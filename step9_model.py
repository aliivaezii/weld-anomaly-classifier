"""
step9_model.py — WeldFusionNet: multimodal MTL architecture.

Purpose
-------
Defines the complete model that ingests sensor, audio, and (optionally)
video data and outputs:
  - 7-class logits  (for defect-type classification)
  - 1 binary logit  (auxiliary, for defect-vs-good training signal)

At inference, the binary prediction is derived from the 7-class softmax:
  p_defect = 1 - softmax[class_good_weld]

Architecture
------------
  Sensor (26, 25) → 1D-CNN Encoder → 64-d
  Audio  (18, 25) → 1D-CNN Encoder → 64-d
  Video  (5, 3, H, W) → MobileNetV3-Small + Temporal Attention → 128-d
                                 ↓
                         Concat → (256-d)
                         FC → BN → ReLU → Dropout → (128-d)
                                 ↓
                 ┌───────────────┴──────────────┐
           7-class head                   Binary head
           Linear(128, 7)                 Linear(128, 1)

Tier 3 upgrade: swap Concat+MLP → Transformer fusion head.

Usage
-----
  Imported by step11_train.py.
  Standalone test:
    python -m pipeline.step9_model --config config.yaml
"""

import argparse
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.utils import load_config

log = logging.getLogger(__name__)

# 7 classes with training data
CLASSES_WITH_DATA = [0, 1, 2, 6, 7, 8, 11]
NUM_CLASSES = 7


# ═══════════════════════════════════════════════════════════════════
#  MODALITY-SPECIFIC ENCODERS
# ═══════════════════════════════════════════════════════════════════

class Conv1dEncoder(nn.Module):
    """
    Lightweight 1D-CNN for short multivariate time series.
    Input:  (B, in_channels, seq_len)   e.g. (B, 26, 25)
    Output: (B, embed_dim)              e.g. (B, 64)
    """

    def __init__(self, in_channels, embed_dim=64, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),       # (B, hidden, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """x: (B, C, T)"""
        h = self.net(x).squeeze(-1)        # (B, hidden)
        return self.fc(h)                  # (B, embed_dim)


class TinyCNNVideoEncoder(nn.Module):
    """
    Fallback video encoder — no pretrained weights, very fast on CPU.
    Input:  (B, n_frames, 3, H, W)
    Output: (B, embed_dim)
    """

    def __init__(self, embed_dim=128):
        super().__init__()
        self.frame_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),       # (B*T, 64, 1, 1)
        )
        self.temporal_pool = nn.Linear(64, 1)  # attention gate
        self.project = nn.Sequential(
            nn.Linear(64, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """x: (B, T, 3, H, W)"""
        B, T = x.shape[:2]
        flat = x.reshape(B * T, *x.shape[2:])          # (B*T, 3, H, W)
        feats = self.frame_cnn(flat).squeeze(-1).squeeze(-1)  # (B*T, 64)
        feats = feats.reshape(B, T, -1)                 # (B, T, 64)

        # Temporal attention pooling
        attn = self.temporal_pool(feats).squeeze(-1)    # (B, T)
        attn = F.softmax(attn, dim=1).unsqueeze(-1)     # (B, T, 1)
        pooled = (feats * attn).sum(dim=1)              # (B, 64)

        return self.project(pooled)                     # (B, embed_dim)


class MobileNetVideoEncoder(nn.Module):
    """
    MobileNetV3-Small pretrained on ImageNet + temporal attention.
    Input:  (B, n_frames, 3, H, W)
    Output: (B, embed_dim)
    """

    def __init__(self, embed_dim=128, freeze_pct=0.8):
        super().__init__()
        try:
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        except ImportError:
            from torchvision.models import mobilenet_v3_small
            backbone = mobilenet_v3_small(pretrained=True)

        # Extract feature layers (everything except final classifier)
        self.features = backbone.features       # output: (B, 576, H', W')
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = 576

        # Freeze first freeze_pct% of feature layers
        layers = list(self.features.children())
        n_freeze = int(len(layers) * freeze_pct)
        for layer in layers[:n_freeze]:
            for p in layer.parameters():
                p.requires_grad = False

        # Temporal attention pooling
        self.attn_gate = nn.Linear(feat_dim, 1)
        self.project = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """x: (B, T, 3, H, W)"""
        B, T = x.shape[:2]
        flat = x.reshape(B * T, *x.shape[2:])           # (B*T, 3, H, W)
        feats = self.features(flat)                      # (B*T, 576, h, w)
        feats = self.pool(feats).squeeze(-1).squeeze(-1) # (B*T, 576)
        feats = feats.reshape(B, T, -1)                  # (B, T, 576)

        # Temporal attention pooling
        attn = self.attn_gate(feats).squeeze(-1)         # (B, T)
        attn = F.softmax(attn, dim=1).unsqueeze(-1)      # (B, T, 1)
        pooled = (feats * attn).sum(dim=1)               # (B, 576)

        return self.project(pooled)                      # (B, embed_dim)


# ═══════════════════════════════════════════════════════════════════
#  FUSION HEADS
# ═══════════════════════════════════════════════════════════════════

class ConcatMLPFusion(nn.Module):
    """
    Simple concatenation + MLP fusion (Tier 2 default).
    Input:  list of embeddings → concat → FC → BN → ReLU → Dropout
    Output: (B, out_dim)
    """

    def __init__(self, in_dim, out_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, *embeddings):
        return self.net(torch.cat(embeddings, dim=1))


class TransformerFusion(nn.Module):
    """
    Tier 3 upgrade: treat each modality embedding as a token,
    run through Transformer layers, classify from [CLS] token.

    Input:  list of embeddings (each (B, D))
    Output: (B, out_dim)
    """

    def __init__(self, token_dim, out_dim=128, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.token_dim = token_dim

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)

        # Positional embeddings (max 5 tokens: [CLS] + up to 4 modalities)
        self.pos_embed = nn.Parameter(torch.randn(1, 5, token_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=n_heads,
            dim_feedforward=token_dim * 2,
            dropout=dropout, activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.project = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, *embeddings):
        """Each embedding: (B, D). All must have the same D = token_dim."""
        B = embeddings[0].shape[0]

        # Stack modality tokens
        tokens = torch.stack(embeddings, dim=1)          # (B, M, D)

        # Prepend [CLS]
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)         # (B, M+1, D)

        # Add positional embeddings
        T = tokens.shape[1]
        tokens = tokens + self.pos_embed[:, :T, :]

        # Transformer
        out = self.transformer(tokens)                   # (B, M+1, D)

        # [CLS] token output
        cls_out = out[:, 0, :]                           # (B, D)
        return self.project(cls_out)                     # (B, out_dim)


# ═══════════════════════════════════════════════════════════════════
#  WELDFUSIONNET — THE COMPLETE MODEL
# ═══════════════════════════════════════════════════════════════════

class WeldFusionNet(nn.Module):
    """
    Multi-Task Learning model for weld defect detection.

    Parameters
    ----------
    num_classes : int        – number of output classes (7)
    use_video : bool         – include video branch
    video_backbone : str     – "mobilenet_v3_small" or "tiny"
    fusion_type : str        – "concat" or "transformer"
    transformer_layers : int – number of transformer layers (Tier 3)
    transformer_heads : int  – number of attention heads (Tier 3)
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        use_video=False,
        video_backbone="mobilenet_v3_small",
        fusion_type="concat",
        transformer_layers=2,
        transformer_heads=4,
    ):
        super().__init__()
        self.use_video = use_video
        self.fusion_type = fusion_type

        # ── Modality encoders ──
        self.sensor_encoder = Conv1dEncoder(in_channels=26, embed_dim=64)
        self.audio_encoder = Conv1dEncoder(in_channels=18, embed_dim=64)

        sensor_dim = 64
        audio_dim = 64
        video_dim = 128

        if use_video:
            if video_backbone == "tiny":
                self.video_encoder = TinyCNNVideoEncoder(embed_dim=video_dim)
            else:
                self.video_encoder = MobileNetVideoEncoder(embed_dim=video_dim)
            total_dim = sensor_dim + audio_dim + video_dim   # 256
        else:
            self.video_encoder = None
            total_dim = sensor_dim + audio_dim               # 128

        # ── Fusion ──
        fusion_out = 128
        if fusion_type == "transformer":
            # For transformer, all tokens must have the same dim.
            # Project everything to a common token_dim.
            token_dim = 128
            self.sensor_proj = nn.Linear(sensor_dim, token_dim)
            self.audio_proj = nn.Linear(audio_dim, token_dim)
            if use_video:
                self.video_proj = nn.Linear(video_dim, token_dim)
            self.fusion = TransformerFusion(
                token_dim=token_dim,
                out_dim=fusion_out,
                n_layers=transformer_layers,
                n_heads=transformer_heads,
            )
        else:
            self.sensor_proj = None
            self.audio_proj = None
            self.fusion = ConcatMLPFusion(in_dim=total_dim, out_dim=fusion_out)

        # ── Prediction heads ──
        self.head_multiclass = nn.Linear(fusion_out, num_classes)
        self.head_binary = nn.Linear(fusion_out, 1)

    def forward(self, sensor, audio, video=None):
        """
        Parameters
        ----------
        sensor : (B, 26, 25)
        audio  : (B, 18, 25)
        video  : (B, T, 3, H, W) or None

        Returns
        -------
        logits_mc : (B, num_classes)  — raw logits for 7-class
        logit_bin : (B, 1)           — raw logit for binary
        """
        s = self.sensor_encoder(sensor)          # (B, 64)
        a = self.audio_encoder(audio)            # (B, 64)

        if self.use_video and video is not None:
            v = self.video_encoder(video)        # (B, 128)
        else:
            v = None

        # Fuse
        if self.fusion_type == "transformer":
            s_proj = self.sensor_proj(s)
            a_proj = self.audio_proj(a)
            if v is not None:
                v_proj = self.video_proj(v)
                fused = self.fusion(s_proj, a_proj, v_proj)
            else:
                fused = self.fusion(s_proj, a_proj)
        else:
            if v is not None:
                fused = self.fusion(s, a, v)
            else:
                fused = self.fusion(s, a)

        logits_mc = self.head_multiclass(fused)  # (B, 7)
        logit_bin = self.head_binary(fused)      # (B, 1)

        return logits_mc, logit_bin

    def count_parameters(self):
        """Return (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ── Factory from config ─────────────────────────────────────────────

def build_model(cfg, use_video=False):
    """Create a WeldFusionNet from the config dict."""
    tcfg = cfg.get("training", {})
    t3 = cfg.get("tier3", {})

    fusion_type = "transformer" if t3.get("transformer_fusion", False) else "concat"

    model = WeldFusionNet(
        num_classes=NUM_CLASSES,
        use_video=use_video,
        video_backbone=tcfg.get("video_backbone", "mobilenet_v3_small"),
        fusion_type=fusion_type,
        transformer_layers=t3.get("transformer_layers", 2),
        transformer_heads=t3.get("transformer_heads", 4),
    )

    total, trainable = model.count_parameters()
    print(f"  Model: WeldFusionNet (fusion={fusion_type}, video={'ON' if use_video else 'OFF'})")
    print(f"  Parameters: {total:,} total, {trainable:,} trainable")

    return model


# ── CLI: architecture summary ───────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 9: Model architecture check")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    cfg = load_config(args.config)

    model = build_model(cfg, use_video=args.video)
    print(f"\n  Architecture:\n{model}\n")

    # Dummy forward pass
    B = 4
    sensor = torch.randn(B, 26, 25)
    audio = torch.randn(B, 18, 25)
    video = torch.randn(B, 5, 3, 224, 224) if args.video else None

    logits_mc, logit_bin = model(sensor, audio, video)
    print(f"  Output shapes: multiclass={logits_mc.shape}, binary={logit_bin.shape}")
    print(f"  ✅ Forward pass OK")
