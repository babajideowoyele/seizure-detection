import numpy as np

# MediaPipe Pose landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
NOSE = 0

# Joint angle triplets (parent, joint, child)
ANGLE_TRIPLETS = [
    (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
    (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
    (NOSE, LEFT_SHOULDER, LEFT_HIP),
    (NOSE, RIGHT_SHOULDER, RIGHT_HIP),
]

# Left-right symmetric pairs
SYMMETRIC_PAIRS = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_ELBOW, RIGHT_ELBOW),
    (LEFT_WRIST, RIGHT_WRIST),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_KNEE, RIGHT_KNEE),
    (LEFT_ANKLE, RIGHT_ANKLE),
]

# Key landmarks for FFT analysis
FFT_LANDMARKS = [LEFT_WRIST, RIGHT_WRIST, LEFT_ANKLE, RIGHT_ANKLE,
                  LEFT_ELBOW, RIGHT_ELBOW, NOSE]
FFT_N_BINS = 5


def compute_velocity(arr: np.ndarray) -> np.ndarray:
    """Frame-to-frame velocity for xyz. Shape: (150, 33, 3)."""
    xyz = arr[:, :, :3]
    vel = np.zeros_like(xyz)
    vel[1:] = xyz[1:] - xyz[:-1]
    return vel


def compute_acceleration(velocity: np.ndarray) -> np.ndarray:
    """Frame-to-frame acceleration from velocity. Shape: (150, 33, 3)."""
    acc = np.zeros_like(velocity)
    acc[1:] = velocity[1:] - velocity[:-1]
    return acc


def compute_speed(velocity: np.ndarray) -> np.ndarray:
    """Per-landmark speed (L2 norm of velocity). Shape: (150, 33)."""
    return np.linalg.norm(velocity, axis=2)


def compute_joint_angles(arr: np.ndarray) -> np.ndarray:
    """Angles at joints defined by ANGLE_TRIPLETS. Shape: (150, 10)."""
    xyz = arr[:, :, :3]
    angles = np.zeros((arr.shape[0], len(ANGLE_TRIPLETS)), dtype=np.float32)

    for i, (a, b, c) in enumerate(ANGLE_TRIPLETS):
        ba = xyz[:, a, :] - xyz[:, b, :]
        bc = xyz[:, c, :] - xyz[:, b, :]
        cos_angle = np.sum(ba * bc, axis=1) / (
            np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-8
        )
        cos_angle = np.clip(cos_angle, -1, 1)
        angles[:, i] = np.arccos(cos_angle) / np.pi  # normalize to [0, 1]

    return angles


def compute_symmetry(arr: np.ndarray) -> np.ndarray:
    """Left-right asymmetry for symmetric pairs. Shape: (150, 6)."""
    xyz = arr[:, :, :3]
    sym = np.zeros((arr.shape[0], len(SYMMETRIC_PAIRS)), dtype=np.float32)

    for i, (left, right) in enumerate(SYMMETRIC_PAIRS):
        diff = xyz[:, left, :] - xyz[:, right, :]
        sym[:, i] = np.linalg.norm(diff, axis=1)

    return sym


def compute_fft_features(arr: np.ndarray) -> np.ndarray:
    """Low-frequency FFT power for key landmarks. Shape: (n_bins * n_landmarks * 3,)."""
    features = []
    for lm_idx in FFT_LANDMARKS:
        for coord in range(3):
            signal = arr[:, lm_idx, coord]
            fft = np.abs(np.fft.rfft(signal))
            power = fft[1:FFT_N_BINS + 1] if len(fft) > FFT_N_BINS else fft[1:]
            features.extend(power.tolist())
    return np.array(features, dtype=np.float32)


def extract_features(arr: np.ndarray, nan_mask: np.ndarray) -> np.ndarray:
    """Full feature extraction pipeline.

    Input: preprocessed (150, 33, 5) array and (150,) nan_mask
    Output: (150, F) feature tensor

    Features per frame:
        Raw xyz, vis, pres:  33*5 = 165
        Velocity xyz:        33*3 = 99
        Acceleration xyz:    33*3 = 99
        Speed magnitude:     33
        Joint angles:        10
        Symmetry:            6
        NaN mask:            1
        FFT power (broadcast): 7*3*5 = 105
    Total: 518
    """
    velocity = compute_velocity(arr)
    acceleration = compute_acceleration(velocity)
    speed = compute_speed(velocity)
    angles = compute_joint_angles(arr)
    symmetry = compute_symmetry(arr)
    fft_feat = compute_fft_features(arr)

    n_frames = arr.shape[0]

    raw_flat = arr.reshape(n_frames, -1)            # (150, 165)
    vel_flat = velocity.reshape(n_frames, -1)        # (150, 99)
    acc_flat = acceleration.reshape(n_frames, -1)    # (150, 99)
    fft_broadcast = np.tile(fft_feat, (n_frames, 1))  # (150, 105)
    nan_col = nan_mask.reshape(n_frames, 1)           # (150, 1)

    combined = np.concatenate([
        raw_flat,       # 165
        vel_flat,       # 99
        acc_flat,       # 99
        speed,          # 33
        angles,         # 10
        symmetry,       # 6
        nan_col,        # 1
        fft_broadcast,  # 105
    ], axis=1)

    return combined  # (150, 518)
