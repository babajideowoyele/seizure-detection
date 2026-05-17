# Video-Based Seizure Detection

Detects pediatric seizures from MediaPipe pose-landmark sequences using a 5-fold ensemble of CNN-GRU and TCN models. Built for the *Video-Based Seizure Detection Challenge*.

Each input is a `(150, 33, 5)` array — 150 frames × 33 MediaPipe Pose landmarks × `(x, y, z, visibility, presence)`. The model outputs a binary label (`1` = seizure, `0` = non-seizure) per segment.

## Approach

| Stage | What happens |
| --- | --- |
| **Preprocess** ([preprocessing.py](preprocessing.py)) | Forward/back-fill all-NaN frames, center on mid-hip, scale by torso length. |
| **Features** ([features.py](features.py)) | 518 per-frame features: raw landmarks (165), velocity (99), acceleration (99), per-landmark speed (33), 10 joint angles, 6 left-right symmetry distances, NaN mask (1), and broadcast FFT power for 7 key landmarks (105). |
| **Models** ([model.py](model.py)) | **CNN-GRU**: 1D-CNN front-end → BiGRU(2) → learned attention pooling → MLP head. **TCN**: 5 dilated residual blocks (dilations 1, 2, 4, 8, 16) → attention pooling → MLP head. |
| **Training** ([train.py](train.py)) | 5-fold `StratifiedGroupKFold` grouped by `child_id` (no subject leakage). `FocalLoss(α=0.75, γ=2.0)`, AdamW, OneCycleLR. Per-fold F1 threshold search. Early stopping on val F1 (patience 25). |
| **Augmentation** ([dataset.py](dataset.py)) | Time reversal, Gaussian noise, frame dropout, same-class mixup. Weighted sampling to balance classes. |
| **Inference** ([inference.py](inference.py)) | Loads all 10 fold checkpoints, runs 7-variant TTA (reverse, jitter, scale ±5%, time shifts ±3), averages probabilities, applies tuned ensemble threshold. |

5-fold CV F1 from [checkpoints/results.json](checkpoints/results.json):

| Model | Mean F1 |
| --- | --- |
| CNN-GRU | 0.628 |
| TCN | 0.635 |
| Ensemble threshold | 0.65 |

## Project layout

```
config.py              Dataclass with all paths & hyperparameters
preprocessing.py       NaN handling + hip-centering + torso normalization
features.py            Kinematic + FFT feature extraction (150 × 518)
dataset.py             Torch Dataset with on-the-fly augmentation
model.py               CNNGRUModel, TCNModel, FocalLoss, Attention
train.py               Full 5-fold × 2-architecture training loop
train_all.py           Local training entrypoint
inference.py           Ensemble loading + TTA + prediction
main.py                Docker entrypoint for batch inference on .npy arrays
video_to_features.py   Video → (150, 33, 5) MediaPipe pose array
predict_video.py       Run the ensemble directly on .mp4/.avi/.mov files
Dockerfile             Reproducible inference image
requirements.txt       numpy, pandas, torch, scikit-learn, mediapipe, opencv
checkpoints/           Saved fold weights + normalization stats + results.json
models/                Place pose_landmarker.task here (not committed)
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.12 (per Dockerfile). Training expects a CUDA GPU; inference runs fine on CPU.

## Training

Update `train_data_dir` in [config.py](config.py#L8) to point at the directory containing `train_data.csv` and the per-segment `.npy` files. Filenames must follow `child_<id>_<segment>.npy` so child IDs can be parsed for grouped CV.

```bash
python train_all.py
```

This will:

1. Preprocess every sample and extract features once (no caching — takes a few minutes).
2. Save normalization stats to `checkpoints/feat_mean.npy` and `feat_std.npy`.
3. Train 10 models (5 folds × 2 architectures), saving each to `checkpoints/`.
4. Write per-fold thresholds and F1 scores to `checkpoints/results.json`.

To tune the ensemble threshold separately, set `"ensemble_threshold"` in `results.json` (currently `0.65`); otherwise inference averages the per-fold thresholds.

## Inference

Local:

```bash
python main.py <test_data_dir> <output_csv>
```

`test_data_dir` should contain `.npy` files matching the same `(150, 33, 5)` shape. Output is a CSV with `segment_name,label` columns.

Docker:

```bash
docker build -t seizure-detection .
docker run --rm \
  -v /path/to/test_data:/data \
  -v /path/to/output:/output \
  -e INPUT="" \
  -e OUTPUT="test_data.csv" \
  seizure-detection
```

The image is also built and pushed to GHCR by [.github/workflows/docker-push.yml](.github/workflows/docker-push.yml) on manual dispatch.

## Inference from raw video

`main.py` only accepts pre-extracted `.npy` arrays (the competition format). To run the same ensemble on raw video files, use [predict_video.py](predict_video.py), which runs MediaPipe Pose to produce a `(150, 33, 5)` array per video (evenly sampling 150 frames across the clip), then routes through the existing TTA + ensemble path.

One-time setup — download the MediaPipe Pose Landmarker model:

```bash
mkdir models
curl -L -o models/pose_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

Override the location with `MEDIAPIPE_MODEL=/some/path/pose_landmarker.task` or by editing `mediapipe_model_path` in [config.py](config.py).

Run on a single video or a directory (mixed `.npy` + video is fine):

```bash
python predict_video.py path/to/clip.mp4 predictions.csv
python predict_video.py path/to/videos/   predictions.csv
```

Output CSV columns: `segment_name, label, probability`.

## Notes

- **No subject leakage**: `StratifiedGroupKFold` groups by `child_id` parsed from the filename, so the same child never appears in both train and val.
- **Class imbalance** is handled in three places: `WeightedRandomSampler`, `FocalLoss`, and per-fold threshold search on F1.
- **Reproducibility**: a single `seed=42` controls torch and numpy; CV folds are deterministic.
