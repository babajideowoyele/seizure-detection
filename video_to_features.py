"""Convert a video file into the (150, 33, 5) MediaPipe pose array the model expects.

Per-landmark channels: (x, y, z, visibility, presence) — matches the
training-data convention. Frames where no pose is detected are left as NaN
so preprocessing.fill_nan_frames can handle them downstream.
"""
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _evenly_spaced_indices(n_total: int, n_target: int) -> np.ndarray:
    """Pick n_target frame indices evenly distributed across [0, n_total).

    If the video has fewer frames than n_target, the last real frame is
    repeated to pad.
    """
    if n_total <= 0:
        return np.zeros(n_target, dtype=int)
    if n_total <= n_target:
        idx = np.arange(n_total)
        pad = np.full(n_target - n_total, n_total - 1, dtype=int)
        return np.concatenate([idx, pad])
    return np.linspace(0, n_total - 1, n_target).round().astype(int)


def _read_selected_frames(
    video_path: Path, frame_indices: np.ndarray
) -> list[np.ndarray | None]:
    """Decode only the frames at the given indices. Returns RGB ndarrays."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    wanted = set(int(i) for i in frame_indices)
    frames_by_idx: dict[int, np.ndarray] = {}
    cur = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if cur in wanted:
            frames_by_idx[cur] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if len(frames_by_idx) == len(wanted):
                break
        cur += 1
    cap.release()

    return [frames_by_idx.get(int(i)) for i in frame_indices]


def video_to_landmarks(
    video_path: Path,
    model_asset_path: Path,
    n_frames: int = 150,
) -> np.ndarray:
    """Extract a (n_frames, 33, 5) pose array from a video file.

    Uses MediaPipe Tasks PoseLandmarker in VIDEO mode. Frames where no pose
    is detected stay as NaN — preprocessing.fill_nan_frames will fill them.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    indices = _evenly_spaced_indices(total, n_frames)
    frames = _read_selected_frames(video_path, indices)

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_asset_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )

    arr = np.full((n_frames, 33, 5), np.nan, dtype=np.float32)

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        last_ts_ms = -1
        for i, (frame, src_idx) in enumerate(zip(frames, indices)):
            if frame is None:
                continue
            ts_ms = int(1000.0 * int(src_idx) / fps)
            # PoseLandmarker requires strictly increasing timestamps; bump if needed
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = landmarker.detect_for_video(mp_image, ts_ms)
            if not result.pose_landmarks:
                continue
            for j, lm in enumerate(result.pose_landmarks[0]):
                arr[i, j, 0] = lm.x
                arr[i, j, 1] = lm.y
                arr[i, j, 2] = lm.z
                arr[i, j, 3] = lm.visibility
                arr[i, j, 4] = lm.presence

    return arr
