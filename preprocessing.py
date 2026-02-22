import numpy as np


def get_nan_mask(arr: np.ndarray) -> np.ndarray:
    """Per-frame binary mask: 1 if frame is all-NaN, 0 otherwise."""
    return np.all(np.isnan(arr[:, :, 0]), axis=1).astype(np.float32)


def fill_nan_frames(arr: np.ndarray) -> np.ndarray:
    """Forward-fill then backward-fill NaN frames.

    If ALL frames are NaN, fill with zeros.
    """
    result = arr.copy()
    n_frames = result.shape[0]
    nan_frames = np.all(np.isnan(result[:, :, 0]), axis=1)

    if nan_frames.all():
        result[:] = 0.0
        return result

    # Forward fill
    last_valid = None
    for i in range(n_frames):
        if not nan_frames[i]:
            last_valid = result[i].copy()
        elif last_valid is not None:
            result[i] = last_valid

    # Backward fill remaining leading NaNs
    first_valid = None
    for i in range(n_frames - 1, -1, -1):
        if not nan_frames[i]:
            first_valid = result[i].copy()
        elif first_valid is not None:
            result[i] = first_valid

    return result


def center_on_hip(arr: np.ndarray) -> np.ndarray:
    """Center x,y,z coordinates relative to mid-hip point.

    MediaPipe landmarks 23 (left hip) and 24 (right hip).
    """
    result = arr.copy()
    mid_hip = (result[:, 23, :3] + result[:, 24, :3]) / 2.0  # (150, 3)
    result[:, :, :3] -= mid_hip[:, np.newaxis, :]
    return result


def normalize_by_torso(arr: np.ndarray) -> np.ndarray:
    """Scale xyz by torso length (mid-shoulder to mid-hip distance)."""
    result = arr.copy()
    mid_shoulder = (result[:, 11, :3] + result[:, 12, :3]) / 2.0
    mid_hip = (result[:, 23, :3] + result[:, 24, :3]) / 2.0
    torso_len = np.linalg.norm(mid_shoulder - mid_hip, axis=1, keepdims=True)  # (150, 1)

    valid_mask = torso_len > 0.01
    median_torso = np.median(torso_len[valid_mask]) if np.any(valid_mask) else 1.0
    torso_len = np.where(torso_len > 0.01, torso_len, median_torso)

    result[:, :, :3] /= torso_len[:, np.newaxis, :]
    return result


def preprocess_sample(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full preprocessing pipeline for a single sample.

    Returns:
        processed: (150, 33, 5) preprocessed array
        nan_mask: (150,) binary NaN frame indicator
    """
    nan_mask = get_nan_mask(arr)
    filled = fill_nan_frames(arr)
    centered = center_on_hip(filled)
    scaled = normalize_by_torso(centered)
    return scaled, nan_mask
