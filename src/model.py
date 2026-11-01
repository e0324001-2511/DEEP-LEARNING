"""FemSense multimodal network: CNN (image) + dense NN (clinical) -> fusion -> 3 heads."""
import torch
import torch.nn as nn
from torchvision import models

from src.config import BACKBONES, DEFAULT_BACKBONE, DISEASES, SEVERITIES, get_img_size


class FemSenseNet(nn.Module):
    """Multimodal classifier with a configurable torchvision image backbone."""

    def __init__(self, n_clinical: int, backbone: str = DEFAULT_BACKBONE,
                 pretrained: bool = False):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone {backbone!r}. Choose from: {', '.join(BACKBONES)}")
        self.backbone_name = backbone
        self.img_size = get_img_size(backbone)
        try:
            self.backbone = models.get_model(
                backbone, weights="DEFAULT" if pretrained else None
            )
        except (OSError, RuntimeError) as exc:
            if pretrained:
                raise RuntimeError(
                    f"Could not load pretrained weights for {backbone}. "
                    "Check your internet connection and try again, or omit --pretrained."
                ) from exc
            raise

        if backbone.startswith("resnet"):
            self.feat_dim = 512 if backbone == "resnet18" else 2048
            self.backbone.fc = nn.Identity()
        elif backbone == "efficientnet_b0":
            self.feat_dim = 1280
            self.backbone.classifier = nn.Identity()
        elif backbone == "densenet121":
            self.feat_dim = 1024
            self.backbone.classifier = nn.Identity()
        elif backbone.startswith("vgg"):
            self.feat_dim = 512
            self.backbone.avgpool = nn.AdaptiveAvgPool2d(1)
            self.backbone.classifier = nn.Identity()
            for layer in self.backbone.modules():
                if isinstance(layer, nn.ReLU):
                    layer.inplace = False
        else:
            self.feat_dim = 768
            self.backbone.heads = nn.Identity()

        # --- Clinical branch: dense network ---
        self.tab = nn.Sequential(
            nn.Linear(n_clinical, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU())
        # --- Multimodal fusion (feature concatenation) ---
        self.fusion = nn.Sequential(nn.Linear(self.feat_dim + 32, 128), nn.ReLU(), nn.Dropout(0.3))
        # --- Task heads ---
        self.disease_head = nn.Linear(128, len(DISEASES))
        self.severity_head = nn.Linear(128, len(SEVERITIES))
        self.risk_head = nn.Linear(128, 1)

    def get_cam_target(self):
        """Return the final spatial/token layer used for Grad-CAM."""
        if self.backbone_name.startswith("resnet"):
            return self.backbone.layer4
        if self.backbone_name == "densenet121":
            return self.backbone.features
        if self.backbone_name.startswith("vgg"):
            return self.backbone.features[-2]
        if self.backbone_name == "efficientnet_b0":
            return self.backbone.features[-1]
        return self.backbone.encoder.layers[-1]

    def forward(self, img, clin):
        z = self.fusion(torch.cat([self.backbone(img), self.tab(clin)], dim=1))
        return self.disease_head(z), self.severity_head(z), self.risk_head(z).squeeze(1)
