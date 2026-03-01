"""
step10_losses.py — Focal Loss + Multi-Task Learning loss combiner.

Purpose
-------
Custom loss functions for the WeldFusionNet MTL training:
  1. FocalLoss — down-weights easy examples, focuses on hard/minority cases
  2. MTLLoss  — weighted combination of FocalLoss (7-class) + BCE (binary)

Why Focal Loss?
  Standard CrossEntropy treats every sample equally.  With 5:1 class imbalance
  (good_weld vs crater_cracks), the model wastes gradient on easy majority
  samples.  Focal Loss with γ=2 gives hard examples 49× more gradient than
  easy ones, which is critical for Macro F1.

Usage
-----
  Imported by step11_train.py.
  Standalone test:
    python -m pipeline.step10_losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

    Parameters
    ----------
    gamma : float          – focusing parameter (default 2.0)
    weight : Tensor or None – per-class weights (same as CE weight)
    reduction : str        – 'mean' or 'sum' or 'none'
    """

    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits, targets):
        """
        logits:  (B, C) raw logits
        targets: (B,)   integer class labels
        """
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.weight, reduction="none",
        )
        pt = torch.exp(-ce_loss)                     # probability of correct class
        focal_weight = (1.0 - pt) ** self.gamma
        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MTLLoss(nn.Module):
    """
    Multi-Task Learning loss combiner.

    total = α × FocalLoss(7-class logits, labels)
          + β × BCEWithLogitsLoss(binary logit, is_defect)

    The binary target is derived from the multi-class label:
      is_defect = (label != 0)    # 0 = good_weld

    Parameters
    ----------
    alpha : float                – weight for multi-class focal loss
    beta : float                 – weight for binary BCE loss
    gamma : float                – focal loss γ parameter
    class_weights : Tensor       – (num_classes,) per-class weights for focal loss
    pos_weight : float or None   – positive class weight for BCE
    """

    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0,
                 class_weights=None, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.focal = FocalLoss(gamma=gamma, weight=class_weights)

        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, logits_mc, logit_bin, labels):
        """
        logits_mc : (B, C)  — raw multi-class logits
        logit_bin : (B, 1)  — raw binary logit
        labels    : (B,)    — integer class labels (0 = good_weld)

        Returns
        -------
        total_loss : scalar
        loss_dict  : {"focal": float, "bce": float, "total": float}
        """
        # Multi-class focal loss
        focal_loss = self.focal(logits_mc, labels)

        # Binary target: 1 = defect, 0 = good
        binary_target = (labels != 0).float().unsqueeze(1)   # (B, 1)
        bce_loss = self.bce(logit_bin, binary_target)

        total = self.alpha * focal_loss + self.beta * bce_loss

        return total, {
            "focal": focal_loss.item(),
            "bce":   bce_loss.item(),
            "total": total.item(),
        }


# ── CLI: quick self-test ────────────────────────────────────────────

if __name__ == "__main__":
    print("  Testing FocalLoss + MTLLoss...")

    B, C = 8, 7
    logits = torch.randn(B, C)
    labels = torch.tensor([0, 1, 2, 6, 7, 8, 11, 0])  # original codes won't work
    # For the model we use contiguous indices 0-6
    labels_idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0])
    logit_bin = torch.randn(B, 1)

    # Test FocalLoss
    fl = FocalLoss(gamma=2.0)
    loss = fl(logits, labels_idx)
    print(f"    FocalLoss: {loss.item():.4f}")

    # Test MTLLoss
    cw = torch.ones(C)
    mtl = MTLLoss(alpha=0.7, beta=0.3, gamma=2.0, class_weights=cw)
    total, details = mtl(logits, logit_bin, labels_idx)
    print(f"    MTLLoss:   total={details['total']:.4f}  "
          f"focal={details['focal']:.4f}  bce={details['bce']:.4f}")

    print("  Losses OK")
