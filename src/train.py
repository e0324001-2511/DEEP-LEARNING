"""Train a FemSense backbone. Usage: python -m src.train --backbone resnet18."""
import argparse
import json
import random

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C
from src.dataset import FemSenseDataset
from src.model import FemSenseNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, device, opt=None):
    train = opt is not None
    model.train(train)
    ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()
    tot_loss = correct = n = 0
    with torch.set_grad_enabled(train):
        for img, clin, disease, severity, risk in loader:
            img, clin, disease, severity, risk = (
                tensor.to(device) for tensor in (img, clin, disease, severity, risk)
            )
            disease_logits, severity_logits, risk_logits = model(img, clin)
            loss = (
                ce(disease_logits, disease)
                + 0.5 * ce(severity_logits, severity)
                + 0.5 * bce(risk_logits, risk)
            )
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot_loss += loss.item() * len(disease)
            correct += (disease_logits.argmax(1) == disease).sum().item()
            n += len(disease)
    return tot_loss / n, correct / n


def main(backbone=C.DEFAULT_BACKBONE, epochs=C.EPOCHS, pretrained=False, lr=None,
         batch_size=None, max_samples=None):
    """Train one backbone and return its best validation disease accuracy."""
    if backbone not in C.BACKBONES:
        raise ValueError(f"Unknown backbone {backbone!r}. Choose from: {', '.join(C.BACKBONES)}")
    set_seed(C.SEED)
    device = torch.device("cpu")
    print("Device:", device)
    if not C.CSV_PATH.exists():
        raise FileNotFoundError("No data found. Run: python -m src.generate_data")
    df = pd.read_csv(C.CSV_PATH)
    if max_samples is not None:
        if max_samples < 8 or max_samples > len(df):
            raise ValueError(f"--max_samples must be between 8 and {len(df)}")
        if max_samples < len(df):
            df, _ = train_test_split(
                df, train_size=max_samples, stratify=df["disease"], random_state=C.SEED
            )

    tr, tmp = train_test_split(
        df, test_size=0.30, stratify=df["disease"], random_state=C.SEED
    )
    va, te = train_test_split(
        tmp, test_size=0.50, stratify=tmp["disease"], random_state=C.SEED
    )
    scaler = StandardScaler().fit(tr[C.CLINICAL_FEATURES].values)
    img_size = C.get_img_size(backbone)
    if batch_size is None:
        batch_size = 16 if backbone in ("resnet101", "vit_b_16") else C.BATCH_SIZE
    if lr is None:
        lr = 1e-4 if pretrained else C.LR

    output_dir = C.OUT_DIR / backbone
    output_dir.mkdir(parents=True, exist_ok=True)
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, C.MODEL_DIR / "scaler.pkl")
    te.to_csv(C.MODEL_DIR / "test_split.csv", index=False)

    tr_l = DataLoader(
        FemSenseDataset(tr, scaler, train=True, img_size=img_size),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    va_l = DataLoader(
        FemSenseDataset(va, scaler, img_size=img_size),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    model = FemSenseNet(len(C.CLINICAL_FEATURES), backbone, pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

    hist = {"tr_loss": [], "va_loss": [], "tr_acc": [], "va_acc": []}
    best = 0.0
    for ep in range(1, epochs + 1):
        tl, ta = run_epoch(model, tr_l, device, opt)
        vl, va_acc = run_epoch(model, va_l, device)
        sched.step()
        for key, value in zip(hist, (tl, vl, ta, va_acc)):
            hist[key].append(value)
        print(
            f"Epoch {ep:02d}/{epochs} | train loss {tl:.3f} acc {ta:.3f} "
            f"| val loss {vl:.3f} acc {va_acc:.3f}"
        )
        if va_acc >= best:
            best = va_acc
            checkpoint = {
                "state_dict": model.state_dict(),
                "n_clinical": len(C.CLINICAL_FEATURES),
                "backbone": backbone,
                "img_size": img_size,
                "pretrained": pretrained,
            }
            checkpoint_path = C.model_path(backbone)
            temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            torch.save(checkpoint, temporary_path)
            temporary_path.replace(checkpoint_path)

    with (output_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(hist, file)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["tr_loss"], label="train")
    ax[0].plot(hist["va_loss"], label="val")
    ax[0].set_title("Loss")
    ax[1].plot(hist["tr_acc"], label="train")
    ax[1].plot(hist["va_acc"], label="val")
    ax[1].set_title("Disease accuracy")
    for axis in ax:
        axis.set_xlabel("epoch")
        axis.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=120)
    plt.close(fig)
    print(f"Best val accuracy: {best:.3f} -> model saved to {C.model_path(backbone)}")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the FemSense multimodal model")
    parser.add_argument("--backbone", choices=C.BACKBONES, default=C.DEFAULT_BACKBONE)
    parser.add_argument("--epochs", type=int, default=C.EPOCHS)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Stratified sample cap for quick smoke tests")
    args = parser.parse_args()
    main(args.backbone, args.epochs, args.pretrained, args.lr, args.batch_size, args.max_samples)
