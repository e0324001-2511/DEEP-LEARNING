"""Train and compare multiple FemSense image backbones."""
import argparse
import gc
import time

import matplotlib
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as C
from src.evaluate import main as evaluate
from src.model import FemSenseNet
from src.train import main as train

RESULT_COLUMNS = [
    "backbone", "params", "train_time", "disease_acc", "macro_f1",
    "precision", "recall", "severity_acc", "risk_mae",
]


def main(backbones=None, epochs=C.EPOCHS, pretrained=False, max_samples=None,
         batch_size=None):
    """Train and evaluate each backbone, keeping going after individual failures."""
    backbones = C.BACKBONES if not backbones or backbones == ["all"] else backbones
    unknown = set(backbones) - set(C.BACKBONES)
    if unknown:
        raise ValueError(f"Unknown backbone(s): {', '.join(sorted(unknown))}")

    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for backbone in backbones:
        print(f"\n=== Comparing {backbone} ===")
        try:
            started = time.perf_counter()
            train(
                backbone=backbone, epochs=epochs, pretrained=pretrained,
                batch_size=batch_size, max_samples=max_samples,
            )
            train_time = time.perf_counter() - started
            metrics = evaluate(backbone)
            model = FemSenseNet(len(C.CLINICAL_FEATURES), backbone, pretrained=False)
            params = sum(parameter.numel() for parameter in model.parameters())
            del model
            rows.append({
                "backbone": backbone,
                "params": params,
                "train_time": round(train_time, 2),
                **metrics,
            })
        except Exception as exc:
            message = f"{backbone}: {type(exc).__name__}: {exc}"
            print(f"FAILED {message}")
            failures.append(message)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(
        C.OUT_DIR / "comparison.csv", index=False
    )
    (C.OUT_DIR / "comparison_failures.log").write_text(
        "\n".join(failures) + ("\n" if failures else ""), encoding="utf-8"
    )
    if rows:
        metrics = ["disease_acc", "macro_f1", "precision", "recall", "severity_acc", "risk_mae"]
        ax = pd.DataFrame(rows).set_index("backbone")[metrics].plot(
            kind="bar", figsize=(13, 6), ylim=(0, 1)
        )
        ax.set_ylabel("Score (lower is better for risk MAE)")
        ax.set_title("FemSense backbone comparison")
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(C.OUT_DIR / "comparison.png", dpi=140)
        plt.close()
    print(f"\nSaved {C.OUT_DIR / 'comparison.csv'} and comparison chart (if any runs passed).")
    if failures:
        print(f"{len(failures)} backbone(s) failed; details are in outputs/comparison_failures.log.")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare FemSense image backbones")
    parser.add_argument("--backbones", nargs="+", default=["all"],
                        help="Backbone names or 'all'")
    parser.add_argument("--epochs", type=int, default=C.EPOCHS)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Stratified sample cap for quick smoke tests")
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()
    main(args.backbones, args.epochs, args.pretrained, args.max_samples, args.batch_size)
