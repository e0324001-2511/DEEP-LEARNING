"""Command-line prediction + explainability on a sample from the dataset.
Usage:  python -m src.predict --index 5 --backbone resnet18
"""
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C
from src.inference import load_model, predict, grad_cam, clinical_importance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0, help="row in data/clinical_data.csv")
    ap.add_argument("--backbone", choices=C.BACKBONES, default=C.DEFAULT_BACKBONE)
    a = ap.parse_args()
    df = pd.read_csv(C.CSV_PATH)
    row = df.iloc[a.index]
    img = Image.open(C.DATA_DIR / row["image_path"]).convert("RGB")
    clin = {k: float(row[k]) for k in C.CLINICAL_FEATURES}

    model, scaler = load_model(backbone=a.backbone)
    res = predict(model, scaler, img, clin)
    print(f"Ground truth : {C.DISEASES[int(row['disease'])]} / {C.SEVERITIES[int(row['severity'])]}")
    print(f"Prediction   : {res['disease']}  (severity: {res['severity']})")
    print(f"Risk score   : {res['risk']:.2f} ({res['risk_level']})")
    print("Advice       :", res["advice"])

    cam = grad_cam(model, img, clin, scaler)
    imp = clinical_importance(model, img, clin, scaler)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].imshow(img); ax[0].set_title("Input image"); ax[0].axis("off")
    ax[1].imshow(img.resize((model.img_size, model.img_size))); ax[1].imshow(cam, cmap="jet", alpha=0.45)
    ax[1].set_title(f"Grad-CAM: {res['disease']}"); ax[1].axis("off")
    items = sorted(imp.items(), key=lambda kv: abs(kv[1]))[-8:]
    ax[2].barh([k for k, _ in items], [v for _, v in items], color=["tab:red" if v > 0 else "tab:blue" for _, v in items])
    ax[2].set_title("Top clinical contributions")
    plt.tight_layout()
    out = C.OUT_DIR / f"explain_{a.index}.png"
    plt.savefig(out, dpi=120)
    print("Saved explanation figure ->", out)


if __name__ == "__main__":
    main()
