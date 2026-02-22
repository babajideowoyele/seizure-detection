import numpy as np
import torch
from torch.utils.data import Dataset


class SeizureDataset(Dataset):
    """Dataset with on-the-fly augmentation for training."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        mixup_alpha: float = 0.0,
    ):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.augment = augment
        self.mixup_alpha = mixup_alpha
        self.seizure_indices = np.where(labels == 1)[0]
        self.non_seizure_indices = np.where(labels == 0)[0]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx].clone()
        y = self.labels[idx].clone()

        if self.augment:
            x, y = self._augment(x, y)

        return x, y

    def _augment(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Time reversal (p=0.3)
        if torch.rand(1).item() < 0.3:
            x = torch.flip(x, [0])

        # Gaussian noise (p=0.5)
        if torch.rand(1).item() < 0.5:
            x = x + torch.randn_like(x) * 0.02

        # Time warp via random resampling (p=0.3)
        if torch.rand(1).item() < 0.3:
            indices = torch.sort(torch.randint(0, x.shape[0], (x.shape[0],)))[0]
            x = x[indices]

        # Random frame dropout (p=0.2)
        if torch.rand(1).item() < 0.2:
            n_drop = torch.randint(1, 15, (1,)).item()
            drop_idx = torch.randint(0, x.shape[0], (n_drop,))
            x[drop_idx] = 0.0

        # Same-class mixup (p=0.3)
        if self.mixup_alpha > 0 and torch.rand(1).item() < 0.3:
            label_val = int(y.item())
            pool = self.seizure_indices if label_val == 1 else self.non_seizure_indices
            mix_idx = np.random.choice(pool)
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            x = lam * x + (1 - lam) * self.features[mix_idx]
            y = torch.tensor(lam * y.item() + (1 - lam) * self.labels[mix_idx].item())

        return x, y


class InferenceDataset(Dataset):
    """Minimal dataset for inference."""

    def __init__(self, features: np.ndarray):
        self.features = torch.FloatTensor(features)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.features[idx]
