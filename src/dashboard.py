"""Reusable dashboard evaluation, multi-model prediction, and report helpers."""
import gc
import io
import time

import numpy as np
import pandas as pd
import torch
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as ReportImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from src import config as C
from src.dataset import FemSenseDataset
from src.inference import load_model, predict


def evaluate_checkpoint(backbone):
    """Evaluate one checkpoint on the persisted held-out test split."""
    split_path = C.MODEL_DIR / "test_split.csv"
    if not split_path.exists():
        raise FileNotFoundError("The saved test split is missing; train a model first.")
    model, scaler = load_model(backbone=backbone, device="cpu")
    frame = pd.read_csv(split_path)
    if frame.empty:
        raise ValueError("The saved test split contains no samples.")
    batch_size = 8 if backbone in ("resnet101", "vit_b_16") else 16
    loader = DataLoader(
        FemSenseDataset(frame, scaler, img_size=model.img_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    disease_true, disease_pred, disease_prob = [], [], []
    severity_true, severity_pred, severity_prob = [], [], []
    risk_true, risk_pred = [], []
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for images, clinical, disease, severity, risk in loader:
            disease_logits, severity_logits, risk_logits = model(images, clinical)
            disease_true.extend(disease.tolist())
            disease_pred.extend(disease_logits.argmax(1).tolist())
            disease_prob.extend(torch.softmax(disease_logits, dim=1).cpu().tolist())
            severity_true.extend(severity.tolist())
            severity_pred.extend(severity_logits.argmax(1).tolist())
            severity_prob.extend(torch.softmax(severity_logits, dim=1).cpu().tolist())
            risk_true.extend(risk.tolist())
            risk_pred.extend(torch.sigmoid(risk_logits).tolist())
    elapsed = time.perf_counter() - started

    precision, recall, f1, _ = precision_recall_fscore_support(
        disease_true,
        disease_pred,
        labels=range(len(C.DISEASES)),
        average="macro",
        zero_division=0,
    )
    try:
        one_hot = label_binarize(disease_true, classes=range(len(C.DISEASES)))
        auc = float(roc_auc_score(
            one_hot, disease_prob, average="macro", multi_class="ovr"
        ))
    except ValueError:
        auc = float("nan")

    result = {
        "backbone": backbone,
        "accuracy": float(accuracy_score(disease_true, disease_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "roc_auc_ovr_macro": auc,
        "severity_accuracy": float(accuracy_score(severity_true, severity_pred)),
        "risk_mae": float(np.mean(np.abs(np.asarray(risk_true) - np.asarray(risk_pred)))),
        "inference_ms_per_image": elapsed * 1000 / len(frame),
        "test_samples": len(frame),
        "disease_true": disease_true,
        "disease_pred": disease_pred,
        "disease_probabilities": disease_prob,
        "severity_true": severity_true,
        "severity_pred": severity_pred,
        "severity_probabilities": severity_prob,
        "risk_true": risk_true,
        "risk_pred": risk_pred,
        "disease_confusion": confusion_matrix(
            disease_true, disease_pred, labels=range(len(C.DISEASES))
        ).tolist(),
        "severity_confusion": confusion_matrix(
            severity_true, severity_pred, labels=range(len(C.SEVERITIES))
        ).tolist(),
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "img_size": model.img_size,
    }
    del model, scaler, loader
    gc.collect()
    return result


def evaluate_ensemble(model_metrics):
    """Measure probability averaging on the shared held-out split."""
    if not model_metrics:
        raise ValueError("No model metrics are available to evaluate the ensemble.")
    disease_probs = np.mean(
        [np.asarray(row["disease_probabilities"]) for row in model_metrics], axis=0
    )
    severity_probs = np.mean(
        [np.asarray(row["severity_probabilities"]) for row in model_metrics], axis=0
    )
    disease_true = model_metrics[0]["disease_true"]
    severity_true = model_metrics[0]["severity_true"]
    risk_true = np.asarray(model_metrics[0]["risk_true"])
    risk_pred = np.mean(
        [np.asarray(row["risk_pred"]) for row in model_metrics], axis=0
    )
    disease_pred = disease_probs.argmax(axis=1).tolist()
    severity_pred = severity_probs.argmax(axis=1).tolist()
    precision, recall, f1, _ = precision_recall_fscore_support(
        disease_true,
        disease_pred,
        labels=range(len(C.DISEASES)),
        average="macro",
        zero_division=0,
    )
    try:
        auc = float(roc_auc_score(
            label_binarize(disease_true, classes=range(len(C.DISEASES))),
            disease_probs,
            average="macro",
            multi_class="ovr",
        ))
    except ValueError:
        auc = float("nan")
    return {
        "backbone": "Probability ensemble",
        "accuracy": float(accuracy_score(disease_true, disease_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "roc_auc_ovr_macro": auc,
        "severity_accuracy": float(accuracy_score(severity_true, severity_pred)),
        "risk_mae": float(np.mean(np.abs(risk_true - risk_pred))),
        "inference_ms_per_image": float(sum(
            row["inference_ms_per_image"] for row in model_metrics
        )),
        "test_samples": len(disease_true),
        "disease_true": disease_true,
        "disease_pred": disease_pred,
        "disease_probabilities": disease_probs.tolist(),
        "severity_true": severity_true,
        "severity_pred": severity_pred,
        "severity_probabilities": severity_probs.tolist(),
        "risk_true": risk_true.tolist(),
        "risk_pred": risk_pred.tolist(),
        "disease_confusion": confusion_matrix(
            disease_true, disease_pred, labels=range(len(C.DISEASES))
        ).tolist(),
        "severity_confusion": confusion_matrix(
            severity_true, severity_pred, labels=range(len(C.SEVERITIES))
        ).tolist(),
        "n_parameters": None,
        "img_size": None,
    }


def run_all_predictions(backbones, pil_image, clinical, on_status=None):
    """Run each requested checkpoint sequentially and isolate per-model failures."""
    results = []
    for backbone in backbones:
        entry = {"backbone": backbone, "status": "Loaded"}
        try:
            model, scaler = load_model(backbone=backbone, device="cpu")
            entry["n_parameters"] = sum(
                parameter.numel() for parameter in model.parameters()
            )
            if on_status:
                on_status(backbone, "Loaded")
                on_status(backbone, "Running")
            started = time.perf_counter()
            entry["prediction"] = predict(model, scaler, pil_image, clinical)
            entry["inference_ms"] = (time.perf_counter() - started) * 1000
            entry["status"] = "Completed"
            if on_status:
                on_status(backbone, "Completed")
        except Exception as exc:
            entry["status"] = "Failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            if on_status:
                on_status(backbone, "Failed")
        finally:
            if "model" in locals():
                del model
            if "scaler" in locals():
                del scaler
            gc.collect()
        results.append(entry)
    return results


def make_ensemble(predictions):
    """Average disease, severity, and risk probabilities from successful models."""
    successful = [
        row for row in predictions
        if row.get("status") == "Completed" and "prediction" in row
    ]
    if not successful:
        return None
    disease = {
        key: float(np.mean([row["prediction"]["disease_probs"][key] for row in successful]))
        for key in C.DISEASES
    }
    severity = {
        key: float(np.mean([row["prediction"]["severity_probs"][key] for row in successful]))
        for key in C.SEVERITIES
    }
    disease_label = max(disease, key=disease.get)
    severity_label = max(severity, key=severity.get)
    labels = [row["prediction"]["disease"] for row in successful]
    agreement = max(labels.count(label) for label in set(labels)) / len(labels)
    risk = float(np.mean([row["prediction"]["risk"] for row in successful]))
    risk_level = (
        "Low" if risk < 0.25 else "Moderate" if risk < 0.55
        else "High" if risk < 0.8 else "Very high"
    )
    return {
        "disease": disease_label,
        "disease_probs": disease,
        "severity": severity_label,
        "severity_probs": severity,
        "risk": risk,
        "risk_level": risk_level,
        "agreement": agreement,
        "model_count": len(successful),
    }


def predictions_csv(predictions, ensemble=None):
    """Return one row per model prediction as downloadable CSV bytes."""
    rows = []
    for entry in predictions:
        row = {
            "backbone": entry["backbone"],
            "status": entry["status"],
            "inference_ms": entry.get("inference_ms"),
            "n_parameters": entry.get("n_parameters"),
            "error": entry.get("error", ""),
        }
        prediction = entry.get("prediction", {})
        row["disease"] = prediction.get("disease")
        row["severity"] = prediction.get("severity")
        row["risk"] = prediction.get("risk")
        for condition, probability in prediction.get("disease_probs", {}).items():
            row[f"disease_prob_{condition}"] = probability
        rows.append(row)
    if ensemble:
        rows.append({
            "backbone": "Probability ensemble",
            "status": "Completed",
            "inference_ms": None,
            "n_parameters": None,
            "error": "",
            "disease": ensemble["disease"],
            "severity": ensemble["severity"],
            "risk": ensemble["risk"],
            **{
                f"disease_prob_{condition}": probability
                for condition, probability in ensemble["disease_probs"].items()
            },
        })
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def comparison_csv(metrics):
    """Return test-set model metrics as downloadable CSV bytes."""
    columns = [
        "backbone", "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "roc_auc_ovr_macro", "severity_accuracy", "risk_mae",
        "inference_ms_per_image", "n_parameters", "test_samples",
    ]
    return pd.DataFrame(metrics)[columns].to_csv(index=False).encode("utf-8")


def make_gradcam_png(pil_image, heatmap, img_size):
    """Render a heatmap overlay to PNG bytes without writing a file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resized = pil_image.convert("RGB").resize((img_size, img_size))
    fig, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(resized)
    axis.imshow(heatmap, cmap="jet", alpha=0.45)
    axis.axis("off")
    output = io.BytesIO()
    fig.savefig(output, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return output.getvalue()


def generate_pdf_report(patient_ref, image_name, clinical, predictions, ensemble,
                        comparison_metrics, explanation=None, explanation_image=None):
    """Create a short, privacy-conscious PDF report in memory."""
    stream = io.BytesIO()
    document = SimpleDocTemplate(
        stream,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title="FemSense AI research decision-support report",
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("FemSense AI — Research Decision-Support Report", styles["Title"]),
        Paragraph(
            "Research prototype only. This report is not a diagnosis, validated "
            "clinical risk assessment, or treatment recommendation.",
            styles["BodyText"],
        ),
        Spacer(1, 10),
        Paragraph(f"Patient reference: {patient_ref}", styles["BodyText"]),
        Paragraph(f"Image reference: {image_name}", styles["BodyText"]),
        Paragraph(
            "Evaluation metrics are computed on the saved held-out test split, "
            "not this patient's inputs.",
            styles["BodyText"],
        ),
        Spacer(1, 12),
        Paragraph("Clinical input summary", styles["Heading2"]),
    ]
    clinical_rows = [["Feature", "Entered value"]]
    clinical_rows.extend([[key, str(value)] for key, value in clinical.items()])
    story.append(Table(clinical_rows, repeatRows=1, hAlign="LEFT"))
    story.extend([Spacer(1, 12), Paragraph("Individual model predictions", styles["Heading2"])])
    prediction_rows = [[
        "Backbone", "Status", "Disease", "Severity", "Risk", "Time (ms)"
    ]]
    for entry in predictions:
        pred = entry.get("prediction", {})
        prediction_rows.append([
            entry["backbone"], entry["status"], pred.get("disease", "—"),
            pred.get("severity", "—"),
            f"{pred['risk']:.1%}" if "risk" in pred else "—",
            f"{entry['inference_ms']:.1f}" if "inference_ms" in entry else "—",
        ])
    model_table = Table(prediction_rows, repeatRows=1, hAlign="LEFT")
    model_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dcefeb")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#aab7b5")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(model_table)
    if ensemble:
        story.extend([
            Spacer(1, 12),
            Paragraph("Probability-averaged ensemble", styles["Heading2"]),
            Paragraph(
                f"Prediction: {ensemble['disease']} "
                f"({ensemble['disease_probs'][ensemble['disease']]:.1%}); "
                f"severity {ensemble['severity']}; risk {ensemble['risk']:.1%}; "
                f"agreement {ensemble['agreement']:.1%}; "
                f"{ensemble['model_count']} contributing models.",
                styles["BodyText"],
            ),
        ])
    if comparison_metrics:
        story.extend([Spacer(1, 12), Paragraph("Held-out test metrics", styles["Heading2"])])
        metric_rows = [[
            "Backbone", "Accuracy", "Macro precision", "Macro recall",
            "Macro F1", "Macro AUC", "Severity acc.", "Inference ms/image",
        ]]
        for item in comparison_metrics:
            auc = item["roc_auc_ovr_macro"]
            metric_rows.append([
                item["backbone"],
                f"{item['accuracy']:.1%}",
                f"{item['precision_macro']:.3f}",
                f"{item['recall_macro']:.3f}",
                f"{item['f1_macro']:.3f}",
                f"{auc:.3f}" if np.isfinite(auc) else "N/A",
                f"{item['severity_accuracy']:.1%}",
                f"{item['inference_ms_per_image']:.1f}",
            ])
        story.append(Table(metric_rows, repeatRows=1, hAlign="LEFT"))
    if explanation:
        story.extend([
            Spacer(1, 12),
            Paragraph("Model explanation (not established medical causes)", styles["Heading2"]),
            Paragraph(explanation, styles["BodyText"]),
        ])
    if explanation_image:
        story.extend([
            Spacer(1, 8),
            ReportImage(io.BytesIO(explanation_image), width=3.5 * inch, height=3.5 * inch),
        ])
    story.extend([
        Spacer(1, 12),
        Paragraph(
            "Limitations: these model outputs depend on the training data and "
            "have not been clinically validated. The bundled dataset is synthetic. "
            "Consult a qualified clinician for medical advice.",
            styles["BodyText"],
        ),
    ])
    document.build(story)
    return stream.getvalue()
