 
"""
losses.py
=========
Loss functions untuk segmentasi multi-kelas dacl10k.

Kombinasi BCE + Dice terbukti efektif untuk dataset
dengan class imbalance tinggi seperti bridge damage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────
#  DICE LOSS
# ─────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Dice Loss untuk segmentasi — robust terhadap class imbalance.

    Args:
        num_classes  : jumlah kelas (19 untuk dacl10k)
        ignore_index : index yang diabaikan (255)
        smooth       : faktor smoothing untuk hindari div/0
    """
    def __init__(
        self,
        num_classes  : int = 19,
        ignore_index : int = 255,
        smooth       : float = 1e-6,
    ):
        super().__init__()
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.smooth       = smooth

    def forward(
        self,
        logits : torch.Tensor,   # (B, C, H, W)
        targets: torch.Tensor,   # (B, H, W) long
    ) -> torch.Tensor:

        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # Buat valid mask (abaikan ignore_index)
        valid = (targets != self.ignore_index)  # (B, H, W)

        # One-hot encode targets
        targets_clamped = targets.clone()
        targets_clamped[~valid] = 0
        targets_oh = F.one_hot(
            targets_clamped, self.num_classes
        ).permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Apply valid mask
        valid_4d   = valid.unsqueeze(1).expand_as(probs)
        probs      = probs * valid_4d
        targets_oh = targets_oh * valid_4d

        # Hitung Dice per kelas
        intersection = (probs * targets_oh).sum(dim=(0, 2, 3))
        cardinality  = (probs + targets_oh).sum(dim=(0, 2, 3))

        dice_per_class = 1.0 - (
            (2.0 * intersection + self.smooth) /
            (cardinality + self.smooth)
        )

        return dice_per_class.mean()


# ─────────────────────────────────────────────
#  FOCAL LOSS
# ─────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss — fokus training pada hard examples.
    Berguna untuk kelas langka seperti ExposedRebars, JointTape.

    Args:
        gamma        : focusing parameter (default 2.0)
        ignore_index : index yang diabaikan
        weight       : class weights tensor (C,)
    """
    def __init__(
        self,
        gamma        : float = 2.0,
        ignore_index : int = 255,
        weight       : Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gamma        = gamma
        self.ignore_index = ignore_index
        self.weight       = weight

    def forward(
        self,
        logits : torch.Tensor,   # (B, C, H, W)
        targets: torch.Tensor,   # (B, H, W) long
    ) -> torch.Tensor:

        ce_loss = F.cross_entropy(
            logits, targets,
            weight       = self.weight,
            ignore_index = self.ignore_index,
            reduction    = "none",
        )  # (B, H, W)

        # Hitung probabilitas kelas yang benar
        probs       = F.softmax(logits, dim=1)
        targets_clamped = targets.clone()
        targets_clamped[targets == self.ignore_index] = 0
        pt = probs.gather(1, targets_clamped.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - pt) ** self.gamma
        focal_loss   = focal_weight * ce_loss

        # Mask ignore pixels
        valid = (targets != self.ignore_index)
        return focal_loss[valid].mean()


# ─────────────────────────────────────────────
#  COMBINED LOSS (DEFAULT)
# ─────────────────────────────────────────────

class SegmentationLoss(nn.Module):
    """
    Loss utama: CrossEntropy + Dice.

    Formula: loss = alpha * CE + (1 - alpha) * Dice

    Ini yang dipakai di semua model untuk fair benchmark.

    Args:
        num_classes  : 19 untuk dacl10k
        ignore_index : 255
        alpha        : bobot CE (default 0.5, Dice juga 0.5)
        class_weights: tensor (C,) untuk class imbalance
        use_focal    : ganti CE dengan Focal Loss
        focal_gamma  : gamma untuk Focal Loss
    """
    def __init__(
        self,
        num_classes   : int = 19,
        ignore_index  : int = 255,
        alpha         : float = 0.5,
        class_weights : Optional[torch.Tensor] = None,
        use_focal     : bool = False,
        focal_gamma   : float = 2.0,
    ):
        super().__init__()
        self.alpha        = alpha
        self.ignore_index = ignore_index

        # Pilih CE atau Focal
        if use_focal:
            self.ce_loss = FocalLoss(
                gamma        = focal_gamma,
                ignore_index = ignore_index,
                weight       = class_weights,
            )
        else:
            self.ce_loss = nn.CrossEntropyLoss(
                weight       = class_weights,
                ignore_index = ignore_index,
            )

        self.dice_loss = DiceLoss(
            num_classes  = num_classes,
            ignore_index = ignore_index,
        )

    def forward(
        self,
        logits : torch.Tensor,   # (B, C, H, W)
        targets: torch.Tensor,   # (B, H, W) long
    ) -> dict:
        """
        Returns dict dengan breakdown loss untuk logging.
        """
        ce   = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        total = self.alpha * ce + (1 - self.alpha) * dice

        return {
            "loss"      : total,
            "loss_ce"   : ce.detach(),
            "loss_dice" : dice.detach(),
        }


# ─────────────────────────────────────────────
#  QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Test SegmentationLoss...")
    B, C, H, W = 2, 19, 512, 512

    logits  = torch.randn(B, C, H, W)
    targets = torch.randint(0, C, (B, H, W))
    targets[0, :10, :10] = 255  # simulasi ignore pixels

    # Test default (CE + Dice)
    criterion = SegmentationLoss(num_classes=C)
    out = criterion(logits, targets)
    print(f"  loss      : {out['loss'].item():.4f}")
    print(f"  loss_ce   : {out['loss_ce'].item():.4f}")
    print(f"  loss_dice : {out['loss_dice'].item():.4f}")

    # Test dengan Focal Loss
    criterion_focal = SegmentationLoss(num_classes=C, use_focal=True)
    out_focal = criterion_focal(logits, targets)
    print(f"\nTest Focal + Dice:")
    print(f"  loss      : {out_focal['loss'].item():.4f}")

    print("\n✅ losses.py OK!")