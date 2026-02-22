import re
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from pathlib import Path

from config import Config
from preprocessing import preprocess_sample
from features import extract_features
from dataset import SeizureDataset
from model import CNNGRUModel, TCNModel, FocalLoss


def parse_child_id(filename: str) -> int:
    """Extract child ID from filename like 'child_123_4.npy'."""
    match = re.match(r"child_(\d+)_\d+\.npy", filename)
    return int(match.group(1))


def load_and_preprocess_all(
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load all training data, preprocess, and extract features."""
    csv_path = config.train_data_dir / "train_data.csv"
    df = pd.read_csv(csv_path)

    all_features = []
    all_labels = []
    all_child_ids = []
    all_filenames = []

    for i, row in df.iterrows():
        filename = row["segment_name"]
        label = row["label"]
        child_id = parse_child_id(filename)

        arr = np.load(config.train_data_dir / filename)
        processed, nan_mask = preprocess_sample(arr)
        feat = extract_features(processed, nan_mask)

        all_features.append(feat)
        all_labels.append(label)
        all_child_ids.append(child_id)
        all_filenames.append(filename)

        if (i + 1) % 200 == 0:
            print(f"  Preprocessed {i + 1}/{len(df)} samples")

    return (
        np.stack(all_features),
        np.array(all_labels, dtype=np.float32),
        np.array(all_child_ids),
        all_filenames,
    )


def get_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Weighted sampler for class balance during training."""
    class_counts = np.bincount(labels.astype(int))
    weights = 1.0 / class_counts[labels.astype(int)]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(labels),
        replacement=True,
    )


def find_optimal_threshold(
    probs: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    """Grid search for threshold that maximizes F1."""
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.20, 0.80, 0.01):
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


def train_one_fold(
    model: nn.Module,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    config: Config,
    device: torch.device,
) -> tuple[nn.Module, float, float]:
    """Train a single model for one fold."""
    train_ds = SeizureDataset(
        train_features, train_labels,
        augment=config.augment_train,
        mixup_alpha=config.mixup_alpha,
    )
    val_ds = SeizureDataset(val_features, val_labels, augment=False)

    sampler = get_sampler(train_labels)
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size * 2)

    criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    steps_per_epoch = len(train_dl)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.lr * 10,
        epochs=config.n_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy="cos",
    )

    best_val_f1 = 0.0
    best_state = None
    best_thresh = 0.5
    patience_counter = 0

    model.to(device)

    for epoch in range(config.n_epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in train_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Validate
        model.eval()
        all_probs = []
        all_labels_val = []
        with torch.no_grad():
            for x_batch, y_batch in val_dl:
                x_batch = x_batch.to(device)
                logits = model(x_batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels_val.extend(y_batch.numpy())

        probs_arr = np.array(all_probs)
        labels_arr = np.array(all_labels_val)
        thresh, f1 = find_optimal_threshold(probs_arr, labels_arr)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch + 1}: loss={epoch_loss / n_batches:.4f}, val_F1={f1:.4f}, thresh={thresh:.3f}")

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_thresh = thresh
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f"    Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    model.cpu()
    return model, best_thresh, best_val_f1


def train_all_models(config: Config) -> None:
    """Full training: 5-fold CV x 2 architectures."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    config.model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading and preprocessing all training data...")
    features, labels, child_ids, filenames = load_and_preprocess_all(config)
    print(f"Features shape: {features.shape}, Labels: {int(labels.sum())}/{len(labels)} seizure")

    # Save normalization stats for inference
    feat_mean = features.mean(axis=(0, 1))
    feat_std = features.std(axis=(0, 1))
    feat_std[feat_std < 1e-8] = 1.0
    np.save(config.model_dir / "feat_mean.npy", feat_mean)
    np.save(config.model_dir / "feat_std.npy", feat_std)

    # Normalize
    features = (features - feat_mean) / feat_std

    # Stratified GroupKFold — balances class ratio AND keeps children together
    gkf = StratifiedGroupKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    results = {"cnn_gru": [], "tcn": []}

    for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(features, labels, groups=child_ids)
    ):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{config.n_folds}")
        print(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")
        train_seizure = int(labels[train_idx].sum())
        val_seizure = int(labels[val_idx].sum())
        print(f"  Train seizure: {train_seizure}/{len(train_idx)} ({train_seizure/len(train_idx):.1%})")
        print(f"  Val seizure: {val_seizure}/{len(val_idx)} ({val_seizure/len(val_idx):.1%})")
        print(f"{'='*60}")

        train_X, train_y = features[train_idx], labels[train_idx]
        val_X, val_y = features[val_idx], labels[val_idx]
        input_dim = features.shape[2]

        # Model A: CNN-GRU
        print("  Training CNN-GRU...")
        model_a = CNNGRUModel(
            input_dim=input_dim, hidden_dim=config.hidden_dim,
            n_layers=config.n_gru_layers, dropout=config.dropout
        )
        model_a, thresh_a, f1_a = train_one_fold(
            model_a, train_X, train_y, val_X, val_y, config, device
        )
        torch.save(model_a.state_dict(), config.model_dir / f"cnn_gru_fold{fold_idx}.pt")
        results["cnn_gru"].append({"threshold": float(thresh_a), "f1": float(f1_a), "fold": fold_idx})
        print(f"  CNN-GRU: F1={f1_a:.4f}, thresh={thresh_a:.3f}")

        # Model B: TCN
        print("  Training TCN...")
        model_b = TCNModel(
            input_dim=input_dim, hidden_dim=config.hidden_dim,
            dropout=config.dropout
        )
        model_b, thresh_b, f1_b = train_one_fold(
            model_b, train_X, train_y, val_X, val_y, config, device
        )
        torch.save(model_b.state_dict(), config.model_dir / f"tcn_fold{fold_idx}.pt")
        results["tcn"].append({"threshold": float(thresh_b), "f1": float(f1_b), "fold": fold_idx})
        print(f"  TCN:     F1={f1_b:.4f}, thresh={thresh_b:.3f}")

    # Save results
    with open(config.model_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    mean_f1_a = np.mean([r["f1"] for r in results["cnn_gru"]])
    mean_f1_b = np.mean([r["f1"] for r in results["tcn"]])
    print(f"\n{'='*60}")
    print(f"Mean CV F1 -- CNN-GRU: {mean_f1_a:.4f}, TCN: {mean_f1_b:.4f}")
    print(f"Combined target: >{max(mean_f1_a, mean_f1_b):.4f}")
    print(f"{'='*60}")
