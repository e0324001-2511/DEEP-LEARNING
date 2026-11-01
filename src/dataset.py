"""Dataset, scaling and image transforms."""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from src.config import DATA_DIR, CLINICAL_FEATURES

RISK_MAP = [0.05, 0.35, 0.65, 0.95]  # severity -> target risk score
_MEAN, _STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_transform(train: bool, img_size: int):
    if train:
        return T.Compose([
            T.Resize((img_size, img_size)), T.RandomHorizontalFlip(),
            T.RandomRotation(10), T.ToTensor(), T.Normalize(_MEAN, _STD)])
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(_MEAN, _STD)])


def preprocess_image(pil_img, img_size: int):
    """PIL image -> normalised tensor (3,H,W) for inference."""
    return build_transform(False, img_size)(pil_img.convert("RGB"))


class FemSenseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, scaler, train: bool = False, img_size: int = 128):
        self.df = df.reset_index(drop=True)
        self.clin = scaler.transform(self.df[CLINICAL_FEATURES].values).astype(np.float32)
        self.tf = build_transform(train, img_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(DATA_DIR / row["image_path"]).convert("RGB")
        sev = int(row["severity"])
        return (self.tf(img), torch.from_numpy(self.clin[i]), int(row["disease"]), sev,
                torch.tensor(RISK_MAP[sev], dtype=torch.float32))
