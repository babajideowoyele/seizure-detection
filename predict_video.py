"""Predict seizure label(s) from raw video file(s) using the trained ensemble.

Usage:
    python predict_video.py <video_or_dir> [output_csv]

Accepts either a single video file or a directory of videos. Mixed
directories containing `.npy` arrays and videos are also supported — `.npy`
files are loaded directly, videos are passed through MediaPipe Pose first.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from config import Config
from inference import load_ensemble, predict_single
from video_to_features import VIDEO_SUFFIXES, video_to_landmarks


def _resolve_model_asset_path(config: Config) -> Path:
    env = os.environ.get("MEDIAPIPE_MODEL")
    candidate = Path(env) if env else config.mediapipe_model_path
    if not candidate.exists():
        raise FileNotFoundError(
            f"MediaPipe pose model not found at {candidate}. "
            "Download pose_landmarker.task from "
            "https://developers.google.com/mediapipe/solutions/vision/pose_landmarker "
            "and set MEDIAPIPE_MODEL or place it at the configured path."
        )
    return candidate


def _gather_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            p for p in path.iterdir()
            if p.suffix.lower() in VIDEO_SUFFIXES or p.suffix.lower() == ".npy"
        )
    raise FileNotFoundError(path)


def _load_array(path: Path, model_asset_path: Path, n_frames: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    return video_to_landmarks(path, model_asset_path, n_frames=n_frames)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("predictions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = Config()
    model_asset_path = _resolve_model_asset_path(config)

    print("=" * 60)
    print("Seizure Detection — Video Inference")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"MediaPipe model: {model_asset_path}")

    models, threshold = load_ensemble(config)
    feat_mean = np.load(config.model_dir / "feat_mean.npy")
    feat_std = np.load(config.model_dir / "feat_std.npy")
    print(f"Loaded {len(models)} models, threshold={threshold:.4f}")

    inputs = _gather_inputs(input_path)
    print(f"Found {len(inputs)} input(s)")

    names: list[str] = []
    labels: list[int] = []
    probs: list[float] = []

    for i, p in enumerate(inputs):
        arr = _load_array(p, model_asset_path, n_frames=config.n_frames)
        prob = predict_single(arr, models, feat_mean, feat_std, use_tta=True)
        label = 1 if prob >= threshold else 0
        names.append(p.name)
        labels.append(label)
        probs.append(prob)
        print(f"  [{i + 1}/{len(inputs)}] {p.name}: prob={prob:.3f} -> {label}")

    df = pd.DataFrame({"segment_name": names, "label": labels, "probability": probs})
    df.to_csv(output_path, index=False)
    print(f"\nWrote {len(df)} predictions to {output_path}")


if __name__ == "__main__":
    main()
