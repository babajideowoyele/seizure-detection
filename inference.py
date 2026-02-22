import json
import numpy as np
import torch
from pathlib import Path

from config import Config
from preprocessing import preprocess_sample
from features import extract_features
from model import CNNGRUModel, TCNModel


def apply_tta(features: np.ndarray) -> list[np.ndarray]:
    """Generate TTA variants of a feature sequence."""
    variants = [features]

    # Time reversal
    variants.append(features[::-1].copy())

    # Small noise
    variants.append(features + np.random.randn(*features.shape).astype(np.float32) * 0.01)

    # Scale up 5%
    variants.append(features * 1.05)

    # Scale down 5%
    variants.append(features * 0.95)

    # Shift time forward
    shifted_fwd = np.roll(features, 3, axis=0)
    shifted_fwd[:3] = features[0]
    variants.append(shifted_fwd)

    # Shift time backward
    shifted_bwd = np.roll(features, -3, axis=0)
    shifted_bwd[-3:] = features[-1]
    variants.append(shifted_bwd)

    return variants


def predict_single(
    arr: np.ndarray,
    models: list[torch.nn.Module],
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    use_tta: bool = True,
) -> float:
    """Predict seizure probability for a single sample."""
    processed, nan_mask = preprocess_sample(arr)
    features = extract_features(processed, nan_mask)

    # Normalize
    features = (features - feat_mean) / feat_std

    variants = apply_tta(features) if use_tta else [features]

    all_probs = []
    for model in models:
        model.eval()
        for feat_var in variants:
            x = torch.FloatTensor(feat_var).unsqueeze(0)
            with torch.no_grad():
                logit = model(x)
                prob = torch.sigmoid(logit).item()
            all_probs.append(prob)

    return float(np.mean(all_probs))


def load_ensemble(config: Config) -> tuple[list[torch.nn.Module], float]:
    """Load all trained models and compute ensemble threshold."""
    models = []

    with open(config.model_dir / "results.json") as f:
        results = json.load(f)

    input_dim = np.load(config.model_dir / "feat_mean.npy").shape[0]

    # Load CNN-GRU folds
    for fold_idx in range(config.n_folds):
        path = config.model_dir / f"cnn_gru_fold{fold_idx}.pt"
        if path.exists():
            model = CNNGRUModel(
                input_dim=input_dim, hidden_dim=config.hidden_dim,
                n_layers=config.n_gru_layers, dropout=0.0,
            )
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            model.eval()
            models.append(model)

    # Load TCN folds
    for fold_idx in range(config.n_folds):
        path = config.model_dir / f"tcn_fold{fold_idx}.pt"
        if path.exists():
            model = TCNModel(
                input_dim=input_dim, hidden_dim=config.hidden_dim, dropout=0.0,
            )
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            model.eval()
            models.append(model)

    # Use optimized ensemble threshold if available, otherwise average per-fold
    if "ensemble_threshold" in results:
        ensemble_thresh = float(results["ensemble_threshold"])
    else:
        all_thresholds = []
        for arch in ["cnn_gru", "tcn"]:
            for r in results.get(arch, []):
                all_thresholds.append(r["threshold"])
        ensemble_thresh = float(np.mean(all_thresholds)) if all_thresholds else 0.5

    return models, ensemble_thresh
