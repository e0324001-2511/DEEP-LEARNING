"""FemSense AI multi-model research dashboard. Run: streamlit run app.py."""
import gc
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize

from src import config as C
from src.dashboard import (
    comparison_csv,
    evaluate_checkpoint,
    evaluate_ensemble,
    generate_pdf_report,
    make_gradcam_png,
    make_ensemble,
    predictions_csv,
    run_all_predictions,
)
from src.inference import clinical_importance, grad_cam, load_model

st.set_page_config(
    page_title="FemSense AI | Multi-model research dashboard",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px;}
    .hero {padding: 1.2rem 1.5rem; border-radius: 16px;
           background: linear-gradient(120deg,#0d4650,#127b78); color: white;
           margin-bottom: 1rem;}
    .disclaimer {padding:.8rem 1rem; border-left:4px solid #e8a24b;
                 background:#fff6e8; color:#4b3b20; border-radius:8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>FemSense AI</h1>'
    "<p>Multimodal research prototype · sequential comparison of trained image backbones</p></div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="disclaimer"><b>Research use only.</b> This software is not a '
    "medical device, diagnosis, validated risk assessment, or treatment recommendation. "
    "Consult a qualified clinician.</div>",
    unsafe_allow_html=True,
)

feature_labels = {
    "age": ("Age", "Years"),
    "bmi": ("BMI", "kg/m²"),
    "cycle_length_days": ("Cycle length", "days"),
    "pelvic_pain_score": ("Pelvic pain score", "score in training data"),
    "dysmenorrhea_score": ("Menstrual pain score", "score in training data"),
    "heavy_bleeding": ("Heavy menstrual bleeding", "0 = no, 1 = yes"),
    "hirsutism_score": ("Excess hair growth score", "score in training data"),
    "acne": ("Acne", "0 = no, 1 = yes"),
    "infertility": ("Infertility history", "0 = no, 1 = yes"),
    "family_history": ("Family history", "0 = no, 1 = yes"),
    "amh": ("AMH", "dataset units"),
    "lh_fsh_ratio": ("LH/FSH ratio", "ratio"),
    "testosterone": ("Testosterone", "dataset units"),
    "ca125": ("CA-125", "dataset units"),
}
binary_features = {"heavy_bleeding", "acne", "infertility", "family_history"}
available_backbones = C.available_backbones()


@st.cache_data(show_spinner=False)
def get_test_metrics(backbones, checkpoint_fingerprint):
    """Cache actual held-out metrics until checkpoints or test data change."""
    rows, failures = [], []
    for backbone in backbones:
        try:
            rows.append(evaluate_checkpoint(backbone))
        except Exception as exc:
            failures.append(f"{backbone}: {type(exc).__name__}: {exc}")
        finally:
            gc.collect()
    ensemble_row = evaluate_ensemble(rows) if rows else None
    return rows, ensemble_row, failures


def metrics_fingerprint(backbones):
    files = [C.MODEL_DIR / "test_split.csv", C.MODEL_DIR / "scaler.pkl"]
    files.extend(C.model_path(backbone) for backbone in backbones)
    return tuple(
        (str(path), path.stat().st_size, path.stat().st_mtime_ns)
        if path.exists() else (str(path), None, None)
        for path in files
    )


def risk_category(probability):
    if probability < 0.25:
        return "Low"
    if probability < 0.50:
        return "Moderate"
    return "High"


if not available_backbones:
    st.error("No trained model checkpoints found. Train models using `python -m src.train`.")
    st.stop()
if not (C.MODEL_DIR / "test_split.csv").exists():
    st.warning("The saved test split is missing. Train a model before reviewing performance metrics.")
    st.stop()
if not (C.MODEL_DIR / "scaler.pkl").exists():
    st.error("The shared clinical scaler is missing. Train a model before running predictions.")
    st.stop()

with st.sidebar:
    st.header("Session")
    st.caption(f"Available checkpoints: {len(available_backbones)} / {len(C.BACKBONES)}")
    st.caption("Patient uploads and inputs are processed in this session and not written to disk.")
    if st.button("Refresh held-out metrics", width="stretch"):
        get_test_metrics.clear()

fingerprint = metrics_fingerprint(available_backbones)
with st.spinner("Evaluating available models on the saved held-out test split..."):
    test_metrics, ensemble_test_metrics, evaluation_failures = get_test_metrics(
        tuple(available_backbones), fingerprint
    )

if not test_metrics:
    st.error("No checkpoint could be evaluated on the saved held-out test split.")
    for message in evaluation_failures:
        st.error(message)
    st.stop()

metrics_frame = pd.DataFrame(test_metrics).sort_values(
    ["accuracy", "f1_macro", "risk_mae"],
    ascending=[False, False, True],
).reset_index(drop=True)
best_backbone = str(metrics_frame.iloc[0]["backbone"])
best_metrics = metrics_frame.iloc[0]
test_count = int(best_metrics["test_samples"])
training_data = pd.read_csv(C.CSV_PATH) if C.CSV_PATH.exists() else None
test_split = pd.read_csv(C.MODEL_DIR / "test_split.csv")

overview_tab, patient_tab, results_tab, comparison_tab, explain_tab, report_tab = st.tabs([
    "Overview", "Patient assessment", "Multi-model results",
    "Performance comparison", "Explainability", "Reports",
])

with overview_tab:
    st.subheader("Model overview")
    overview = st.columns(4)
    overview[0].metric("Available models", len(test_metrics))
    overview[1].metric("Best held-out accuracy", f"{best_metrics['accuracy']:.1%}")
    overview[2].metric("Highest-ranked model", best_backbone)
    overview[3].metric("Held-out samples", test_count)
    st.info(
        "The bundled project data is generated by the synthetic-data demo pipeline. "
        "Unless replaced with documented, appropriately consented data, treat all "
        "evaluation numbers as synthetic/unverified—not clinical performance."
    )
    if training_data is not None and "disease" in training_data:
        counts = training_data["disease"].value_counts().reindex(
            range(len(C.DISEASES)), fill_value=0
        )
        class_frame = pd.DataFrame({
            "Condition": C.DISEASES,
            "Training records": counts.values,
        })
        st.plotly_chart(
            px.bar(class_frame, x="Condition", y="Training records",
                   title="Training dataset class distribution", color="Condition"),
            width="stretch",
        )
        if (counts < 10).any():
            st.warning("At least one disease class has fewer than 10 records in the dataset.")
    st.markdown(
        "**Pipeline:** medical image + 14 clinical features → image backbone + clinical "
        "dense branch → disease, severity, and risk outputs."
    )

with patient_tab:
    st.subheader("Patient data upload")
    st.caption(
        "Provide one image and the existing 14 clinical features. Input files remain "
        "in memory for this session; the app does not save them."
    )
    uploaded_image = st.file_uploader(
        "Upload ultrasound or MRI image (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        key="patient_image",
    )
    image_modality = st.radio(
        "Image modality (descriptive only; not a model input)",
        ["Ultrasound", "MRI"],
        horizontal=True,
    )
    image = None
    image_name = ""
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image).convert("RGB")
            image_name = uploaded_image.name
            st.image(image, caption="Uploaded image preview", width=360)
        except (UnidentifiedImageError, OSError) as exc:
            st.error(f"Could not read the uploaded image: {exc}")

    input_mode = st.radio(
        "Clinical input method", ["Enter values", "Upload clinical CSV"],
        horizontal=True,
    )
    csv_frame = None
    selected_row = None
    if input_mode == "Upload clinical CSV":
        clinical_file = st.file_uploader(
            "Clinical CSV (must include all 14 model features)",
            type=["csv"],
            key="clinical_csv",
        )
        if clinical_file is not None:
            try:
                csv_frame = pd.read_csv(clinical_file)
                missing_columns = [
                    feature for feature in C.CLINICAL_FEATURES
                    if feature not in csv_frame.columns
                ]
                if missing_columns:
                    st.error("CSV is missing required columns: " + ", ".join(missing_columns))
                    csv_frame = None
                elif csv_frame.empty:
                    st.error("The uploaded clinical CSV has no patient rows.")
                    csv_frame = None
                else:
                    selected_index = st.selectbox(
                        "Select patient row",
                        options=list(range(len(csv_frame))),
                        format_func=lambda index: f"Row {index + 1}",
                    )
                    selected_row = csv_frame.iloc[selected_index]
                    st.success("All required clinical features are present.")
            except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
                st.error(f"Could not read the clinical CSV: {exc}")

    clinical = {}
    with st.form("clinical_form", clear_on_submit=False):
        st.markdown("#### Patient reference (optional; do not enter a name)")
        patient_ref = st.text_input("Non-identifying reference", max_chars=48)
        if input_mode == "Enter values":
            st.markdown("#### Demographics and symptoms")
            cols = st.columns(2)
            for index, feature in enumerate(C.CLINICAL_FEATURES):
                label, units = feature_labels[feature]
                help_text = f"{label}; {units}. Enter a value comparable to the training dataset."
                base = 0.0
                if training_data is not None:
                    base = float(training_data[feature].median())
                with cols[index % 2]:
                    if feature in binary_features:
                        clinical[feature] = float(st.selectbox(
                            label, [0, 1], index=0, key=f"manual_{feature}",
                            help=help_text,
                        ))
                    else:
                        clinical[feature] = float(st.number_input(
                            f"{label} ({units})",
                            value=base,
                            step=0.1,
                            min_value=0.0,
                            key=f"manual_{feature}",
                            help=help_text,
                        ))
        else:
            st.caption("Clinical values will be read from the selected CSV row.")
        run_clicked = st.form_submit_button(
            "RUN ALL MODELS", type="primary", width="stretch"
        )

    if input_mode == "Upload clinical CSV" and selected_row is not None:
        try:
            clinical = {
                feature: float(pd.to_numeric(selected_row[feature], errors="raise"))
                for feature in C.CLINICAL_FEATURES
            }
        except (TypeError, ValueError) as exc:
            st.error(f"The selected CSV row contains a non-numeric clinical value: {exc}")
            clinical = {}

    if run_clicked:
        if image is None:
            st.error("Upload a valid PNG, JPG, or JPEG image before running predictions.")
        elif len(clinical) != len(C.CLINICAL_FEATURES):
            st.error("Provide a complete set of the 14 required clinical features.")
        elif not np.isfinite(np.asarray(list(clinical.values()), dtype=float)).all():
            st.error("Clinical inputs must be finite numeric values; missing values are not accepted.")
        else:
            if training_data is not None:
                out_of_range = [
                    feature for feature, value in clinical.items()
                    if feature in training_data
                    and (
                        value < float(training_data[feature].min())
                        or value > float(training_data[feature].max())
                    )
                ]
                if out_of_range:
                    st.warning(
                        "These values are outside the training-data range: "
                        + ", ".join(feature_labels[name][0] for name in out_of_range)
                        + ". The model can still run, but its predictions may be unreliable."
                    )
            status_area = st.empty()
            status = {name: "Waiting" for name in available_backbones}

            def update_status(name, value):
                status[name] = value
                status_area.dataframe(
                    pd.DataFrame([
                        {"Backbone": model_name, "Status": state}
                        for model_name, state in status.items()
                    ]),
                    hide_index=True,
                    width="stretch",
                )

            with st.spinner("Running checkpoints sequentially on CPU..."):
                predictions = run_all_predictions(
                    available_backbones, image, clinical, on_status=update_status
                )
            ensemble = make_ensemble(predictions)
            st.session_state["patient_run"] = {
                "patient_ref": patient_ref.strip() or "Not provided",
                "image_name": f"{image_modality}: {image_name}",
                "clinical": clinical,
                "predictions": predictions,
                "ensemble": ensemble,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.success(
                f"Completed {sum(row['status'] == 'Completed' for row in predictions)} "
                f"of {len(predictions)} model(s)."
            )

with results_tab:
    st.subheader("Individual model predictions")
    patient_run = st.session_state.get("patient_run")
    if not patient_run:
        st.info("Upload patient inputs and click RUN ALL MODELS to view results.")
    else:
        completed = [
            row for row in patient_run["predictions"]
            if row["status"] == "Completed"
        ]
        best_entry = next(
            (row for row in completed if row["backbone"] == best_backbone), None
        )
        if best_entry:
            best_prediction = best_entry["prediction"]
            st.markdown(f"### Highest held-out accuracy: {best_backbone}")
            st.caption(
                f"Test accuracy {best_metrics['accuracy']:.1%} on {test_count} samples. "
                "The patient-specific output below is still an unvalidated research prediction."
            )
            best_columns = st.columns(4)
            best_columns[0].metric(
                "Predicted condition", best_prediction["disease"],
                f"{max(best_prediction['disease_probs'].values()):.1%} model confidence",
            )
            best_columns[1].metric("Severity", best_prediction["severity"])
            best_columns[2].metric("Risk score", f"{best_prediction['risk']:.1%}")
            best_columns[3].metric(
                "Inference", f"{best_entry['inference_ms']:.1f} ms"
            )
        if patient_run["ensemble"]:
            ensemble = patient_run["ensemble"]
            st.markdown("### FemSense AI ensemble prediction")
            ensemble_cols = st.columns(4)
            ensemble_cols[0].metric("Combined condition", ensemble["disease"])
            ensemble_cols[1].metric(
                "Condition probability",
                f"{ensemble['disease_probs'][ensemble['disease']]:.1%}",
            )
            ensemble_cols[2].metric("Combined severity", ensemble["severity"])
            ensemble_cols[3].metric("Combined risk score", f"{ensemble['risk']:.1%}")
            st.caption(
                f"Model agreement: {ensemble['agreement']:.1%} · "
                f"{ensemble['model_count']} contributing model(s). "
                "The probability average is not established as superior to individual models."
            )
            st.plotly_chart(
                px.bar(
                    x=list(ensemble["disease_probs"].keys()),
                    y=list(ensemble["disease_probs"].values()),
                    labels={"x": "Condition", "y": "Mean predicted probability"},
                    title="Ensemble disease probabilities",
                    color=list(ensemble["disease_probs"].keys()),
                ),
                width="stretch",
            )
        for entry in patient_run["predictions"]:
            with st.expander(
                f"{entry['backbone']} — {entry['status']}", expanded=False
            ):
                if entry["status"] != "Completed":
                    st.error(entry.get("error", "Model could not complete prediction."))
                    continue
                prediction = entry["prediction"]
                details = st.columns(4)
                details[0].metric("Disease", prediction["disease"])
                details[1].metric("Severity", prediction["severity"])
                details[2].metric("Risk score", f"{prediction['risk']:.1%}")
                details[3].metric("Inference time", f"{entry['inference_ms']:.1f} ms")
                st.caption(f"Parameters: {entry['n_parameters']:,}")
                st.plotly_chart(
                    px.bar(
                        x=list(prediction["disease_probs"].keys()),
                        y=list(prediction["disease_probs"].values()),
                        labels={"x": "Condition", "y": "Probability"},
                        title=f"{entry['backbone']} disease probabilities",
                    ),
                    width="stretch",
                )
        if completed:
            st.markdown("### Disease probability risk indicators")
            st.caption(
                "Low / Moderate / High are illustrative probability bands only; "
                "they are not clinically validated risk thresholds or individual disease risk."
            )
            risk_rows = []
            for disease in C.DISEASES:
                probability = patient_run["ensemble"]["disease_probs"][disease]
                risk_rows.append({
                    "Condition": disease,
                    "Model probability": probability,
                    "Illustrative band": risk_category(probability),
                })
            st.dataframe(
                pd.DataFrame(risk_rows).style.format({"Model probability": "{:.1%}"}),
                hide_index=True,
                width="stretch",
            )

with comparison_tab:
    st.subheader("DEEP LEARNING MODEL PERFORMANCE COMPARISON")
    st.caption(
        f"Metrics are recomputed from the saved test split ({test_count} samples), "
        "never from the current patient's input."
    )
    if test_count < 100:
        st.warning(
            f"The held-out set contains only {test_count} samples. This is a small, "
            "preliminary evaluation and is not clinical evidence."
        )
    if evaluation_failures:
        for message in evaluation_failures:
            st.warning(message)
    shown_metrics = metrics_frame[[
        "backbone", "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "roc_auc_ovr_macro", "severity_accuracy", "risk_mae",
        "inference_ms_per_image", "n_parameters", "test_samples",
    ]].copy()
    if ensemble_test_metrics:
        shown_metrics = pd.concat(
            [shown_metrics, pd.DataFrame([ensemble_test_metrics])],
            ignore_index=True,
        )
    st.dataframe(
        shown_metrics.rename(columns={
            "backbone": "Model",
            "accuracy": "Accuracy",
            "precision_macro": "Precision (macro)",
            "recall_macro": "Recall (macro)",
            "f1_macro": "F1 (macro)",
            "roc_auc_ovr_macro": "ROC-AUC (macro OVR)",
            "severity_accuracy": "Severity accuracy",
            "risk_mae": "Risk MAE",
            "inference_ms_per_image": "Inference ms / image",
            "n_parameters": "Parameters",
            "test_samples": "Test samples",
        }).style.format({
            "Accuracy": "{:.1%}",
            "Precision (macro)": "{:.3f}",
            "Recall (macro)": "{:.3f}",
            "F1 (macro)": "{:.3f}",
            "ROC-AUC (macro OVR)": "{:.3f}",
            "Severity accuracy": "{:.1%}",
            "Risk MAE": "{:.3f}",
            "Inference ms / image": "{:.1f}",
            "Parameters": "{:,.0f}",
        }, na_rep="N/A"),
        hide_index=True,
        width="stretch",
    )
    chart_columns = st.columns(2)
    for index, (metric, label) in enumerate([
        ("accuracy", "Accuracy"),
        ("precision_macro", "Macro precision"),
        ("recall_macro", "Macro recall"),
        ("f1_macro", "Macro F1"),
        ("inference_ms_per_image", "Inference time (ms/image)"),
    ]):
        best_row = (
            metrics_frame.loc[metrics_frame[metric].idxmax()]
            if metric != "inference_ms_per_image"
            else metrics_frame.loc[metrics_frame[metric].idxmin()]
        )
        fig = px.bar(
            metrics_frame,
            x="backbone",
            y=metric,
            title=f"{label} · best: {best_row['backbone']} ({best_row[metric]:.3g})",
            color="backbone",
        )
        chart_columns[index % 2].plotly_chart(fig, width="stretch")
    roc_model = st.selectbox(
        "Confusion matrix / ROC model",
        options=[row["backbone"] for row in test_metrics]
        + (["Probability ensemble"] if ensemble_test_metrics else []),
    )
    selected_metrics = (
        ensemble_test_metrics if roc_model == "Probability ensemble"
        else next(row for row in test_metrics if row["backbone"] == roc_model)
    )
    conf_columns = st.columns(2)
    for axis, key, title, labels in [
        (conf_columns[0], "disease_confusion", "Disease confusion matrix", C.DISEASES),
        (conf_columns[1], "severity_confusion", "Severity confusion matrix", C.SEVERITIES),
    ]:
        axis.plotly_chart(
            go.Figure(go.Heatmap(
                z=selected_metrics[key],
                x=labels,
                y=labels,
                text=selected_metrics[key],
                texttemplate="%{text}",
                colorscale="Teal",
            )).update_layout(title=title, xaxis_title="Predicted", yaxis_title="True"),
            width="stretch",
        )
    try:
        true_binary = label_binarize(
            selected_metrics["disease_true"], classes=range(len(C.DISEASES))
        )
        roc_fig = go.Figure()
        for class_index, label in enumerate(C.DISEASES):
            fpr, tpr, _ = roc_curve(
                true_binary[:, class_index],
                np.asarray(selected_metrics["disease_probabilities"])[:, class_index],
            )
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, name=label))
        roc_fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Chance",
            line={"dash": "dash", "color": "gray"},
        ))
        roc_fig.update_layout(
            title=f"One-vs-rest ROC curves · {roc_model}",
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
        )
        st.plotly_chart(roc_fig, width="stretch")
    except (ValueError, IndexError):
        st.info("ROC curves are unavailable because the test split lacks class support.")

with explain_tab:
    st.subheader("Explainable AI")
    patient_run = st.session_state.get("patient_run")
    if not patient_run:
        st.info("Run all models for a patient before requesting explanations.")
    else:
        completed = [
            item for item in patient_run["predictions"]
            if item["status"] == "Completed"
        ]
        if completed:
            xai_model = st.selectbox(
                "Choose model to explain",
                [item["backbone"] for item in completed],
                key="xai_backbone",
            )
            if st.button("Generate image and clinical explanations"):
                image_upload = st.session_state.get("patient_image")
                if image_upload is None:
                    st.error("The image must remain available in this session to explain it.")
                else:
                    try:
                        xai_image = Image.open(image_upload).convert("RGB")
                        with st.spinner(f"Generating explanation for {xai_model}..."):
                            model, scaler = load_model(backbone=xai_model)
                            attribution = clinical_importance(
                                model, xai_image, patient_run["clinical"], scaler
                            )
                            heatmap = grad_cam(
                                model, xai_image, patient_run["clinical"], scaler
                            )
                            st.session_state["explanation"] = {
                                "backbone": xai_model,
                                "attribution": attribution,
                                "heatmap": heatmap,
                                "img_size": model.img_size,
                            }
                            del model, scaler
                            gc.collect()
                    except Exception as exc:
                        st.error(f"Could not generate model explanation: {type(exc).__name__}: {exc}")
        explanation = st.session_state.get("explanation")
        if explanation:
            st.warning(
                "These are model attribution visualizations, not medically established "
                "causes or clinical reasoning."
            )
            image_upload = st.session_state.get("patient_image")
            if image_upload is not None:
                original = Image.open(image_upload).convert("RGB")
                first, second = st.columns(2)
                first.image(original, caption="Original input image", width="stretch")
                resized = original.resize((explanation["img_size"], explanation["img_size"]))
                overlay_fig = go.Figure()
                overlay_fig.add_trace(go.Image(z=np.asarray(resized)))
                overlay_fig.add_trace(go.Heatmap(
                    z=explanation["heatmap"], colorscale="Jet", opacity=0.45,
                    showscale=False,
                ))
                overlay_fig.update_layout(
                    title=f"{explanation['backbone']} Grad-CAM / patch-token map",
                    xaxis={"visible": False}, yaxis={"visible": False},
                    margin={"l": 0, "r": 0, "t": 40, "b": 0},
                )
                second.plotly_chart(overlay_fig, width="stretch")
                if explanation["backbone"] == "vit_b_16":
                    st.caption(
                        "ViT explanation uses the final encoder's patch-token grid "
                        "with the CLS token omitted; it is not CNN feature-map Grad-CAM."
                    )
            contribution_frame = pd.DataFrame(
                explanation["attribution"].items(),
                columns=["Clinical feature", "Gradient × input"],
            ).assign(
                Magnitude=lambda frame: frame["Gradient × input"].abs()
            ).nlargest(10, "Magnitude")
            st.plotly_chart(
                px.bar(
                    contribution_frame.sort_values("Gradient × input"),
                    x="Gradient × input", y="Clinical feature",
                    orientation="h", color="Gradient × input",
                    title="Most influential clinical inputs for this model output",
                ),
                width="stretch",
            )

with report_tab:
    st.subheader("Reports and downloads")
    st.download_button(
        "Download held-out model comparison CSV",
        data=comparison_csv(test_metrics + ([ensemble_test_metrics] if ensemble_test_metrics else [])),
        file_name="femsense_heldout_model_comparison.csv",
        mime="text/csv",
    )
    patient_run = st.session_state.get("patient_run")
    if patient_run:
        st.download_button(
            "Download individual prediction results CSV",
            data=predictions_csv(patient_run["predictions"], patient_run["ensemble"]),
            file_name="femsense_patient_model_predictions.csv",
            mime="text/csv",
        )
        explanation = st.session_state.get("explanation")
        explanation_text = ""
        if explanation:
            sorted_features = sorted(
                explanation["attribution"].items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:5]
            explanation_text = (
                f"Explained model: {explanation['backbone']}. "
                "Top absolute gradient-times-input features: "
                + ", ".join(f"{key} ({value:.4g})" for key, value in sorted_features)
                + ". This is model attribution, not an established medical cause."
            )
        explanation_image = None
        image_upload = st.session_state.get("patient_image")
        if explanation and image_upload is not None:
            explanation_image = make_gradcam_png(
                Image.open(image_upload).convert("RGB"),
                explanation["heatmap"],
                explanation["img_size"],
            )
        pdf = generate_pdf_report(
            patient_run["patient_ref"],
            patient_run["image_name"],
            patient_run["clinical"],
            patient_run["predictions"],
            patient_run["ensemble"],
            test_metrics + ([ensemble_test_metrics] if ensemble_test_metrics else []),
            explanation_text,
            explanation_image,
        )
        st.download_button(
            "Download research prediction report (PDF)",
            data=pdf,
            file_name="femsense_research_prediction_report.pdf",
            mime="application/pdf",
        )
    else:
        st.info("Run a patient assessment to enable the PDF and patient prediction CSV.")
    st.caption(
        "Downloads are generated in memory. The app does not write uploaded images, "
        "clinical CSVs, or patient reports to project files."
    )
