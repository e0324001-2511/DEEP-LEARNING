"""Evaluate one backbone on the shared held-out split."""
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C
from src.dataset import FemSenseDataset
from src.inference import load_model


def main(backbone=C.DEFAULT_BACKBONE):
    """Evaluate the selected backbone and return its requested metrics."""
    model, scaler = load_model(backbone=backbone, device="cpu")
    test_path = C.MODEL_DIR / "test_split.csv"
    if not test_path.exists():
        raise FileNotFoundError("Test split not found. Run: python -m src.train")
    te = pd.read_csv(test_path)
    batch_size = 16 if model.backbone_name in ("resnet101", "vit_b_16") else C.BATCH_SIZE
    loader = DataLoader(
        FemSenseDataset(te, scaler, img_size=model.img_size),
        batch_size=batch_size, num_workers=0,
    )
    yd, pred_d, ys, pred_s, risks_true, risks_pred = [], [], [], [], [], []
    with torch.no_grad():
        for img, clin, disease, severity, risk in loader:
            ld, ls, lr = model(img, clin)
            yd.extend(disease.tolist())
            pred_d.extend(ld.argmax(1).tolist())
            ys.extend(severity.tolist())
            pred_s.extend(ls.argmax(1).tolist())
            risks_true.extend(risk.tolist())
            risks_pred.extend(torch.sigmoid(lr).tolist())

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        yd, pred_d, labels=range(len(C.DISEASES)), average="macro", zero_division=0
    )
    metrics = {
        "disease_acc": accuracy_score(yd, pred_d),
        "macro_f1": macro_f1,
        "precision": macro_precision,
        "recall": macro_recall,
        "severity_acc": accuracy_score(ys, pred_s),
        "risk_mae": float(np.mean(np.abs(np.array(risks_true) - np.array(risks_pred)))),
    }
    lines = [
        f"Backbone: {backbone}",
        f"Test samples: {len(yd)}",
        f"Disease accuracy : {metrics['disease_acc']:.3f}",
        f"Macro F1         : {metrics['macro_f1']:.3f}",
        f"Macro precision  : {metrics['precision']:.3f}",
        f"Macro recall     : {metrics['recall']:.3f}",
        f"Severity accuracy: {metrics['severity_acc']:.3f}",
        f"Risk-score MAE   : {metrics['risk_mae']:.3f}",
        "",
        "== Disease report ==",
        classification_report(
            yd, pred_d, labels=range(len(C.DISEASES)),
            target_names=C.DISEASES, zero_division=0,
        ),
        "== Severity report ==",
        classification_report(
            ys, pred_s, labels=range(len(C.SEVERITIES)),
            target_names=C.SEVERITIES, zero_division=0,
        ),
    ]
    text = "\n".join(lines)
    print(text)

    output_dir = C.OUT_DIR / backbone
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.txt").write_text(text, encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(
        confusion_matrix(yd, pred_d, labels=range(4)), display_labels=C.DISEASES
    ).plot(ax=axes[0], colorbar=False, xticks_rotation=30)
    axes[0].set_title("Disease")
    ConfusionMatrixDisplay(
        confusion_matrix(ys, pred_s, labels=range(4)), display_labels=C.SEVERITIES
    ).plot(ax=axes[1], colorbar=False)
    axes[1].set_title("Severity")
    plt.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png", dpi=120)
    plt.close(fig)
    print(f"Saved outputs/{backbone}/metrics.txt and confusion_matrices.png")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a FemSense backbone")
    parser.add_argument("--backbone", choices=C.BACKBONES, default=C.DEFAULT_BACKBONE)
    main(parser.parse_args().backbone)
