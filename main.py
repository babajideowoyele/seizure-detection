"""Docker entrypoint for seizure detection inference.

Usage: python3 main.py <test_data_dir> <output_dir>
"""
import sys
import glob
import numpy as np
import pandas as pd
from pathlib import Path

from config import Config
from inference import load_ensemble, predict_single


def main() -> None:
    test_data_dir = Path(sys.argv[1])
    output_data_dir = Path(sys.argv[2])
    output_data_dir.mkdir(parents=True, exist_ok=True)

    config = Config()

    print("=" * 60)
    print("Seizure Detection Challenge - Inference")
    print("=" * 60)
    print(f"Input directory: {test_data_dir}")
    print(f"Output directory: {output_data_dir}")

    # Load ensemble
    models, threshold = load_ensemble(config)
    feat_mean = np.load(config.model_dir / "feat_mean.npy")
    feat_std = np.load(config.model_dir / "feat_std.npy")

    print(f"Loaded {len(models)} models, threshold={threshold:.4f}")

    # Find all test .npy files
    npy_paths = sorted(glob.glob(str(test_data_dir / "*.npy")))
    print(f"Found {len(npy_paths)} test samples")

    file_names = []
    labels = []

    for i, path in enumerate(npy_paths):
        name = Path(path).name
        arr = np.load(path)

        prob = predict_single(arr, models, feat_mean, feat_std, use_tta=True)
        label = 1 if prob >= threshold else 0

        file_names.append(name)
        labels.append(label)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(npy_paths)} files")

    # Save output
    df = pd.DataFrame({"segment_name": file_names, "label": labels})
    output_path = output_data_dir / "test_data.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} predictions to {output_path}")
    print(f"Predicted seizures: {sum(labels)} ({sum(labels)/len(labels):.1%})")


if __name__ == "__main__":
    main()
