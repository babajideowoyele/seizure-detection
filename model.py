import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for class imbalance."""

    def __init__(self, alpha: float = 0.7, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class Attention(nn.Module):
    """Learned attention pooling over temporal dimension."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attn(x)  # (batch, seq_len, 1)
        weights = F.softmax(weights, dim=1)
        return (x * weights).sum(dim=1)  # (batch, hidden_dim)


class CNNGRUModel(nn.Module):
    """1D-CNN -> BiGRU -> Attention -> Classifier."""

    def __init__(
        self,
        input_dim: int = 518,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.attention = Attention(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 150, input_dim)
        x = x.permute(0, 2, 1)     # (batch, input_dim, 150)
        x = self.cnn(x)            # (batch, 128, 150)
        x = x.permute(0, 2, 1)     # (batch, 150, 128)
        x, _ = self.gru(x)         # (batch, 150, hidden*2)
        x = self.attention(x)      # (batch, hidden*2)
        return self.classifier(x).squeeze(-1)


class TemporalBlock(nn.Module):
    """Dilated convolution block with residual connection."""

    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        return F.relu(out + res)


class TCNModel(nn.Module):
    """Multi-scale Temporal Convolutional Network."""

    def __init__(
        self,
        input_dim: int = 518,
        hidden_dim: int = 128,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            TemporalBlock(input_dim, hidden_dim, kernel_size=3, dilation=1, dropout=dropout),
            TemporalBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2, dropout=dropout),
            TemporalBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=4, dropout=dropout),
            TemporalBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=8, dropout=dropout),
            TemporalBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=16, dropout=dropout),
        ])
        self.attention = Attention(hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 150, input_dim)
        x = x.permute(0, 2, 1)  # (batch, input_dim, 150)
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 1)  # (batch, 150, hidden_dim)
        x = self.attention(x)   # (batch, hidden_dim)
        return self.classifier(x).squeeze(-1)
