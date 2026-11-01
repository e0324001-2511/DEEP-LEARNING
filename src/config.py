"""Central configuration for FemSense AI."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMG_DIR = DATA_DIR / "images"
CSV_PATH = DATA_DIR / "clinical_data.csv"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
MODEL_PATH = MODEL_DIR / "femsense_best.pt"

BACKBONES = [
    "resnet18", "resnet50", "resnet101", "efficientnet_b0",
    "densenet121", "vgg16", "vgg19", "vit_b_16",
]
DEFAULT_BACKBONE = "resnet18"

DISEASES = ["Normal", "PCOS", "Endometriosis", "Adenomyosis"]
SEVERITIES = ["None", "Mild", "Moderate", "Severe"]

# Clinical features fed into the dense network (order matters)
CLINICAL_FEATURES = [
    "age", "bmi", "cycle_length_days", "pelvic_pain_score", "dysmenorrhea_score",
    "heavy_bleeding", "hirsutism_score", "acne", "infertility", "family_history",
    "amh", "lh_fsh_ratio", "testosterone", "ca125",
]

def get_img_size(backbone: str) -> int:
    """Return the input size expected by a backbone."""
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone {backbone!r}. Choose from: {', '.join(BACKBONES)}")
    return 224 if backbone == "vit_b_16" else 128


def model_path(backbone: str) -> Path:
    """Return the backbone-specific checkpoint path."""
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone {backbone!r}. Choose from: {', '.join(BACKBONES)}")
    return MODEL_DIR / f"femsense_{backbone}.pt"


def available_backbones() -> list[str]:
    """List trained backbones, including the original legacy ResNet18 checkpoint."""
    available = [name for name in BACKBONES if model_path(name).exists()]
    if DEFAULT_BACKBONE not in available and MODEL_PATH.exists():
        available.insert(0, DEFAULT_BACKBONE)
    return available


BATCH_SIZE = 32
EPOCHS = 12
LR = 1e-3
SEED = 42
# Set True to use ImageNet-pretrained ResNet18 (needs internet the first time).
USE_PRETRAINED = False
