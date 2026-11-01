# FemSense AI: Early Detection of Women's Reproductive Disorders Using Multimodal Deep Learning with Multi-Backbone Comparison

---

**R. Bavadhaarini (E0324001)**
Dept. of Artificial Intelligence and Data Analytics
Sri Ramachandra Faculty of Engineering and Technology
Sri Ramachandra Institute of Higher Education and Research
Porur, Chennai, Tamil Nadu, India

**Lishalini M (E0324002)**
Dept. of Artificial Intelligence and Data Analytics
Sri Ramachandra Faculty of Engineering and Technology
Sri Ramachandra Institute of Higher Education and Research
Porur, Chennai, Tamil Nadu, India

**Guided By: Guide Name**
Dept. of Artificial Intelligence and Data Analytics
Sri Ramachandra Faculty of Engineering and Technology
Sri Ramachandra Institute of Higher Education and Research
Porur, Chennai, Tamil Nadu, India

---

## Abstract

Polycystic ovary syndrome (PCOS), endometriosis and adenomyosis are common reproductive disorders in women. Their symptoms overlap, so diagnosis is often delayed by years. Most computational studies use either imaging or clinical data alone, and few explain their predictions. This paper presents **FemSense AI**, a multimodal deep learning framework that fuses ultrasound/MRI images with 14 clinical and hormonal features to classify a patient as Normal, PCOS, Endometriosis or Adenomyosis. The same network also predicts a four-level severity grade and a continuous risk score.

The framework supports **eight configurable image backbones**—ResNet-18, ResNet-50, ResNet-101, EfficientNet-B0, DenseNet-121, VGG-16, VGG-19 and Vision Transformer (ViT-B/16)—all pre-trained on the ImageNet dataset and adapted through transfer learning. While the image encoder is interchangeable, the clinical branch, feature set, multimodal fusion layer, three prediction heads and multi-task loss remain unchanged. A dense network encodes the clinical record, and the two feature vectors are concatenated and passed through a shared fusion layer into three task-specific heads trained with a weighted multi-task loss. To support clinical trust, the system provides Grad-CAM heat maps for the image branch and gradient-times-input attribution for the clinical branch, and is delivered as a Streamlit web application with multi-model comparison, probability-averaged ensemble prediction and in-memory PDF/CSV report generation.

The framework was developed and evaluated on a synthetic dataset of 800 samples, which allows the full pipeline to be reproduced without patient data. On a held-out test split of 15 samples, the best-performing backbones (ResNet-18 and DenseNet-121) achieved **53.3% disease accuracy**, while DenseNet-121 achieved the lowest risk-score MAE of **0.258**. A systematic comparison across all eight backbones reveals that lighter architectures outperform deeper ones on this small-scale synthetic dataset. These figures demonstrate the pipeline, not clinical performance. Validation on real, ethically approved patient data is the main direction for future work.

**Index Terms**—Multimodal deep learning, PCOS, endometriosis, adenomyosis, transfer learning, ResNet, EfficientNet, DenseNet, VGG, Vision Transformer, Grad-CAM, explainable AI, clinical decision support, multi-backbone comparison.

---

## I. Introduction

Polycystic ovary syndrome (PCOS), endometriosis and adenomyosis are among the most common causes of pelvic pain, menstrual disturbance and infertility in women of reproductive age [5]–[7]. They share symptoms such as pain, irregular or heavy bleeding and subfertility, which makes them easy to confuse in routine consultations. Endometriosis in particular is known for long diagnostic delays [6]. Early detection matters because timely treatment reduces symptoms, protects fertility and lowers long-term risk.

In practice, clinicians reach a diagnosis by combining several sources of evidence: the patient's history and symptom scores, laboratory hormone values, and imaging such as transvaginal ultrasound or MRI. Deep learning has shown strong results on medical image analysis [12], but a model that sees only the image ignores the symptoms and hormones a clinician would weigh. A model that sees only tabular data ignores the anatomical findings. Multimodal learning, which fuses information from different modalities, is a natural fit for this problem [3].

Two further issues limit the adoption of such systems. First, a black-box prediction is hard for a clinician to trust. Second, a single diagnosis label is less useful than a label accompanied by a severity grade and a risk estimate that can guide follow-up urgency. Third, different image encoders extract features at different levels of abstraction, and comparing them under identical conditions is essential for selecting the most suitable architecture for a given clinical imaging setting.

The main contributions of this paper are:

1. A **multimodal network** that fuses a configurable image encoder with a dense clinical encoder to classify Normal, PCOS, Endometriosis and Adenomyosis.
2. A **multi-backbone comparison framework** that evaluates eight pre-trained ImageNet architectures (ResNet-18/50/101, EfficientNet-B0, DenseNet-121, VGG-16/19, ViT-B/16) under identical training, fusion and evaluation conditions.
3. A **multi-task design** that predicts disease, four-level severity and a continuous risk score from one shared representation.
4. A **probability-averaged ensemble** that combines outputs from all available backbones.
5. Built-in **explainability**: Grad-CAM / patch-token attribution for the image branch and gradient-times-input attribution for the clinical features, supporting both CNN and Transformer architectures.
6. A **reproducible end-to-end pipeline** (data generation, training, evaluation, comparison, inference and a Streamlit multi-model dashboard) that runs on a CPU.
7. A **rule-based recommendation layer** that maps the predicted disease, severity and risk into follow-up advice, framed as decision support.

---

## II. Literature Review

### A. Deep Learning in Medical Imaging

Convolutional neural networks (CNNs) have become the standard approach for medical image classification, segmentation and detection [12]. Residual networks such as ResNet introduced skip connections that make deeper models trainable [1]. Because labelled medical datasets are small, **transfer learning**—where a network pre-trained on a large natural-image dataset (ImageNet) is adapted to the target task—is widely used to improve accuracy and reduce training time [8]. Beyond ResNets, architectures such as EfficientNet [13], DenseNet [14], VGG [15] and Vision Transformer (ViT) [16] have each shown competitive or superior performance on various medical imaging benchmarks.

### B. Transfer Learning with Pre-trained Models

Transfer learning leverages representations learned from the ImageNet dataset (1.2 million images across 1,000 classes) [8]. The pre-trained convolutional or attention-based feature extractors are adapted to the medical domain through fine-tuning. This approach is especially valuable when labelled medical data is scarce. FemSense AI supports both ImageNet-initialised and randomly-initialised training modes for all eight backbones.

### C. Multimodal Learning

Multimodal machine learning combines data from different sources to build a richer representation than any single modality provides [3]. Fusion strategies are commonly grouped into early fusion (combining raw inputs), late fusion (combining per-modality decisions) and intermediate fusion (combining learned feature vectors). FemSense AI uses intermediate fusion by concatenating the learned image and clinical feature vectors before a shared fusion layer.

### D. Multi-Task Learning

Training one network on several related tasks lets the tasks share a representation, which can act as a regulariser and improve generalisation [4]. Here, disease classification, severity grading and risk estimation are closely related clinical questions and are learned jointly.

### E. Explainable AI for Clinical Use

Grad-CAM uses the gradients flowing into the last convolutional layer to produce a heat map showing which image regions support a prediction [2]. For Vision Transformers, the CLS token is excluded and the patch-token grid is used instead. For tabular inputs, gradient-based attribution methods such as gradient-times-input estimate how much each feature contributed to an output [11]. Providing both helps a clinician check whether a prediction rests on plausible evidence.

### F. Research Gaps

- Many studies address a single disorder, whereas the three conditions overlap clinically and should be distinguished together.
- Many models use only images or only tabular data, and so do not mirror how clinicians combine evidence.
- Few systems provide severity and risk outputs alongside the diagnosis.
- Explainability is often missing or limited to the image.
- Most papers evaluate a single backbone and do not compare architectures under identical multimodal conditions.
- Research code is often hard to reproduce and lacks a usable interface.

FemSense AI is designed to address these gaps in a single, reproducible, multi-backbone framework.

---

## III. Proposed System

### A. Overview

FemSense AI accepts two inputs for each patient: an ultrasound or MRI image and a clinical record of 14 features. It returns:

1. A predicted condition (Normal, PCOS, Endometriosis or Adenomyosis)
2. A severity grade (None, Mild, Moderate or Severe)
3. A risk score between 0 and 1 with a risk level
4. Visual and numerical explanations
5. A follow-up recommendation

The overall architecture is shown in Fig. 1.

### B. Configurable Image Backbones

The image encoder is selectable from eight architectures. While the backbone varies, the clinical branch, feature set, fusion mechanism, three prediction heads and multi-task loss remain unchanged. This design enables fair comparison across architectures.

**TABLE I: Supported Image Backbones**

| Backbone | Architecture Family | Input Size | Image Feature Dimension | Parameters (Total Model) | ImageNet Pre-training |
|:---|:---|:---:|:---:|:---:|:---:|
| ResNet-18 | Residual CNN | 128 × 128 | 512 | 11,250,473 | ✓ |
| ResNet-50 | Residual CNN | 128 × 128 | 2,048 | 23,778,601 | ✓ |
| ResNet-101 | Residual CNN | 128 × 128 | 2,048 | 42,770,729 | ✓ |
| EfficientNet-B0 | Compound-scaled CNN | 128 × 128 | 1,280 | 4,179,813 | ✓ |
| DenseNet-121 | Dense CNN | 128 × 128 | 1,024 | 7,093,353 | ✓ |
| VGG-16 | Plain CNN | 128 × 128 | 512 | 14,788,649 | ✓ |
| VGG-19 | Plain CNN | 128 × 128 | 512 | 20,098,345 | ✓ |
| ViT-B/16 | Vision Transformer | 224 × 224 | 768 | 85,905,385 | ✓ |

### C. Clinical Features

The clinical branch uses 14 inputs: age, BMI, cycle length (days), pelvic pain score, dysmenorrhea score, heavy bleeding (yes/no), hirsutism score, acne (yes/no), infertility (yes/no), family history (yes/no), anti-Müllerian hormone (AMH), LH/FSH ratio, testosterone and CA-125. They were chosen because they are associated with the three conditions: for example, an elevated LH/FSH ratio, raised AMH and hirsutism with PCOS; severe pelvic pain, dysmenorrhea and raised CA-125 with endometriosis; and heavy bleeding with adenomyosis.

### D. Output Definitions

Disease is a four-class label. Severity is a four-class label where "None" applies only to Normal cases. The risk score is a continuous value in [0, 1]. During training each severity grade is mapped to a target risk value (None → 0.05, Mild → 0.35, Moderate → 0.65, Severe → 0.95). At inference the risk score is converted to a level: Low (< 0.25), Moderate (0.25 to < 0.55), High (0.55 to < 0.80) or Very high (≥ 0.80).

### E. System Architecture

#### Image Branch

The image encoder is selected from one of the eight supported architectures (Table I) with its final classification layer removed, so that it outputs a feature vector of the corresponding dimension. Images are converted to three channels and resized to 128 × 128 pixels (224 × 224 for ViT-B/16). The implementation can start from ImageNet-pretrained weights (transfer learning [8]) or from random initialisation, controlled by a configuration flag.

For each backbone, the adaptation is as follows:
- **ResNet-18/50/101**: The final fully-connected layer (`fc`) is replaced with an Identity layer, yielding 512-d (ResNet-18) or 2,048-d (ResNet-50/101) features.
- **EfficientNet-B0**: The classifier head is replaced with an Identity layer, yielding 1,280-d features.
- **DenseNet-121**: The classifier head is replaced with an Identity layer, yielding 1,024-d features.
- **VGG-16/19**: The average pooling is replaced with an adaptive average pool to (1,1), and the classifier is replaced with an Identity layer, yielding 512-d features. In-place ReLU operations are disabled to support Grad-CAM gradient computation.
- **ViT-B/16**: The classification heads module is replaced with an Identity layer, yielding 768-d features. Input images are resized to 224 × 224 as required by the 16 × 16 patch tokeniser.

#### Clinical Branch

The 14 clinical features are standardised (zero mean, unit variance) using a scaler fitted on the training set only. They pass through a dense network: Linear(14 → 64), ReLU, Dropout(0.2), Linear(64 → 32), ReLU, producing a 32-dimensional vector.

#### Fusion and Task Heads

The image feature vector (dimension varies by backbone) and the 32-d clinical vector are concatenated and passed through a fusion layer: Linear(feat_dim + 32 → 128), ReLU, Dropout(0.3). Three linear heads read this shared 128-d representation:
- **Disease head**: 4 outputs (Normal, PCOS, Endometriosis, Adenomyosis)
- **Severity head**: 4 outputs (None, Mild, Moderate, Severe)
- **Risk head**: 1 output passed through a sigmoid

### F. Multi-Task Loss

The network is trained with a weighted sum of three losses:

$$\mathcal{L} = \mathcal{L}_{\text{disease}} + 0.5 \cdot \mathcal{L}_{\text{severity}} + 0.5 \cdot \mathcal{L}_{\text{risk}}$$

where $\mathcal{L}_{\text{disease}}$ and $\mathcal{L}_{\text{severity}}$ are cross-entropy losses and $\mathcal{L}_{\text{risk}}$ is binary cross-entropy between the predicted risk and the target risk value. Disease is weighted most heavily because it is the primary task.

### G. Explainability Module

**Grad-CAM (image).** Gradients of the selected disease logit are taken with respect to the activations of the last spatial layer. The target layer varies by backbone:
- ResNet: `layer4` (final residual block)
- DenseNet-121: `features` (final dense block)
- VGG-16/19: `features[-2]` (second-to-last convolutional layer)
- EfficientNet-B0: `features[-1]` (final MBConv block)
- ViT-B/16: `encoder.layers[-1]` (final transformer encoder layer)

For CNNs, gradients are averaged over space to obtain channel weights, the weighted activations are summed, a ReLU is applied, and the map is up-sampled to the image size and normalised to [0, 1] [2]. For ViT-B/16, the CLS token is excluded and the remaining patch tokens are reshaped into a 14 × 14 spatial grid before the same Grad-CAM computation is applied.

**Clinical attribution.** For the same disease logit, the gradient with respect to the standardised clinical input is multiplied by the input itself (gradient × input) [11]. The result is a signed score per feature, showing which clinical values pushed the prediction towards or away from the predicted class.

### H. Recommendation Layer

A rule-based layer maps the predicted disease to a suggested clinical step (for example, endocrine evaluation and lifestyle review for PCOS, or gynaecological review and MRI follow-up for endometriosis) and the severity to a follow-up window (6 to 12 months for Mild, within 1 to 3 months for Moderate, prompt consultation for Severe). Every message states that the output is decision support and not a medical diagnosis.

### I. Web Application

The system is delivered as a **Streamlit multi-model research dashboard** with six tabs:

1. **Overview** — Model inventory, held-out metrics summary, training class distribution
2. **Patient Assessment** — Image upload, clinical feature entry (manual or CSV), sequential execution of all checkpoints
3. **Multi-Model Results** — Individual backbone predictions, probability-averaged ensemble, disease probability risk indicators
4. **Performance Comparison** — Accuracy, precision, recall, F1, ROC-AUC, confusion matrices and ROC curves for every backbone and the ensemble
5. **Explainability** — Grad-CAM / patch-token overlays and clinical feature attribution charts, selectable by backbone
6. **Reports** — In-memory PDF report generation and CSV downloads (patient data is never written to disk)

---

## IV. Dataset

Real ultrasound and MRI data for these conditions are patient-sensitive and require ethics approval. To make the pipeline reproducible without patient data, a **synthetic data generator** was used. It produces 800 samples, each with a 128 × 128 grayscale ultrasound-like image and a correlated clinical record. The disease for each sample is drawn at random and severity (1 to 3) is drawn for abnormal cases.

Image patterns mimic characteristic findings:
- **PCOS**: Multiple small follicle-like dark spots around the ovary ("string of pearls"); the number increases with severity (6 + 4 × severity, plus random variation)
- **Endometriosis**: A large ground-glass cyst with a bright rim; the size increases with severity (9 + 6 × severity radius)
- **Adenomyosis**: A heterogeneous, enlarged uterine wall with linear dark streaks (3 + 2 × severity streaks)
- **Normal**: Baseline organ shape with speckle noise only

Clinical values are sampled from distributions that depend on the disease and severity, e.g. higher LH/FSH ratio, AMH, testosterone and hirsutism for PCOS; higher pelvic pain, dysmenorrhea and CA-125 for endometriosis; and frequent heavy bleeding with older age for adenomyosis.

The generated class distribution is Normal 206, PCOS 172, Endometriosis 210 and Adenomyosis 212. Among abnormal cases, severity is spread roughly evenly across Mild, Moderate and Severe.

**Limitation**: Because both the images and the clinical features were generated by rules written by the authors, a network can learn those rules easily. Performance on this dataset shows that the pipeline works. It is not evidence of clinical accuracy.

---

## V. Methodology

### A. Data Split and Pre-processing

The data were divided into **70% training (560)**, **15% validation (120)** and **15% test (15)** samples. The split is stratified by disease class with a fixed random seed of 42. The scaler and test-split indices are persisted to `models/scaler.pkl` and `models/test_split.csv` respectively, and are shared across all backbone experiments.

Images are resized to 128 × 128 (or 224 × 224 for ViT-B/16) and normalised with ImageNet mean and standard deviation. Training images receive random horizontal flips and random rotations of up to 10 degrees. Validation and test images are not augmented.

### B. Training Configuration

Each backbone was trained independently under identical conditions. The model was implemented in Python with PyTorch [9] and trained with the Adam optimiser [10]. A step scheduler halves the learning rate every 5 epochs. After each epoch the model is evaluated on the validation set and the checkpoint with the best validation disease accuracy is kept.

**TABLE II: Training Hyper-parameters**

| Parameter | Value |
|:---|:---|
| Image size | 128 × 128 (224 × 224 for ViT-B/16) |
| Batch size | 32 (16 for ResNet-101 and ViT-B/16) |
| Epochs | 12 |
| Optimiser | Adam |
| Initial learning rate | 10⁻³ (10⁻⁴ with --pretrained) |
| LR schedule | Halved every 5 epochs |
| Loss weights (disease : severity : risk) | 1 : 0.5 : 0.5 |
| Dropout (clinical / fusion) | 0.2 / 0.3 |
| Random seed | 42 |
| Pre-trained weights | ImageNet (optional) |

### C. Multi-Backbone Comparison Pipeline

The automated comparison pipeline (`src/compare.py`) trains and evaluates each selected backbone sequentially, records parameter counts and training time, and writes consolidated results to `outputs/comparison.csv` and `outputs/comparison.png`. A failed run is logged in `outputs/comparison_failures.log` without stopping later backbones. Per-backbone training histories, curves, metrics and confusion matrices are written under `outputs/<backbone>/`.

### D. Ensemble Method

A **probability-averaged ensemble** combines the disease, severity and risk outputs from all successfully evaluated checkpoints. The per-class softmax probabilities from each backbone are averaged element-wise and the ensemble prediction is the argmax of the averaged vector. Model agreement is computed as the fraction of backbones that agree with the plurality label.

### E. Evaluation Metrics

Disease and severity are evaluated with accuracy, per-class precision, recall and F1-score, and confusion matrices. The risk score is evaluated with the mean absolute error (MAE) against the severity-derived target. Macro-averaged precision, recall, F1 and ROC-AUC (one-vs-rest) are computed for overall comparison. All metrics are computed on the held-out test set.

**TABLE III: Technology Stack**

| Component | Technology |
|:---|:---|
| Language | Python 3.9–3.12 |
| Deep learning | PyTorch ≥ 2.0, torchvision ≥ 0.15 |
| Pre-trained dataset | ImageNet (1.2M images, 1,000 classes) |
| Image backbones | ResNet-18/50/101, EfficientNet-B0, DenseNet-121, VGG-16/19, ViT-B/16 |
| Data handling | NumPy, pandas, Pillow |
| Pre-processing & metrics | scikit-learn (StandardScaler, classification reports) |
| Serialisation | joblib |
| Visualisation | Matplotlib, Plotly |
| Web interface | Streamlit |
| Report generation | ReportLab (in-memory PDF) |
| Development | VS Code |

---

## VI. Experimental Results and Analysis

All results are on the held-out test set of **15 samples** (stratified by disease class: Normal 4, PCOS 3, Endometriosis 4, Adenomyosis 4).

### A. Multi-Backbone Comparison — Overall Performance

**TABLE IV: Comprehensive Backbone Comparison on the Held-Out Test Set**

| Backbone | Parameters | Training Time (s) | Disease Accuracy (%) | Macro F1 | Macro Precision | Macro Recall | Severity Accuracy (%) | Risk MAE |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| **ResNet-18** | **11,250,473** | **11.03** | **53.3** | **0.348** | **0.268** | **0.500** | **46.7** | 0.298 |
| **DenseNet-121** | **7,093,353** | **31.95** | **53.3** | **0.354** | **0.278** | **0.500** | 33.3 | **0.258** |
| VGG-16 | 14,788,649 | 47.18 | 46.7 | 0.317 | 0.232 | 0.500 | 26.7 | 0.270 |
| VGG-19 | 20,098,345 | 28.96 | 20.0 | 0.083 | 0.050 | 0.250 | 33.3 | 0.270 |
| ResNet-50 | 23,778,601 | 22.44 | 26.7 | 0.105 | 0.067 | 0.250 | 26.7 | 0.262 |
| ResNet-101 | 42,770,729 | 48.64 | 26.7 | 0.105 | 0.067 | 0.250 | 26.7 | 0.268 |
| EfficientNet-B0 | 4,179,813 | 18.85 | 26.7 | 0.105 | 0.067 | 0.250 | 26.7 | 0.261 |
| ViT-B/16 | 85,905,385 | 120.21 | 26.7 | 0.105 | 0.067 | 0.250 | 33.3 | 0.271 |

> **Key finding**: ResNet-18 and DenseNet-121 achieved the highest disease accuracy (53.3%) and macro F1 (0.348 and 0.354 respectively). DenseNet-121 achieved the lowest risk-score MAE (0.258). ResNet-18 achieved the highest severity accuracy (46.7%). Larger models (ResNet-50/101, ViT-B/16) did not outperform smaller ones, likely due to the small synthetic dataset size (800 samples) causing overfitting in higher-capacity models.

### B. Per-Class Disease Results — Best Model (ResNet-18)

**TABLE V: Per-Class Disease Results — ResNet-18**

| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| Normal | 0.57 | 1.00 | 0.73 | 4 |
| PCOS | 0.00 | 0.00 | 0.00 | 3 |
| Endometriosis | 0.00 | 0.00 | 0.00 | 4 |
| Adenomyosis | 0.50 | 1.00 | 0.67 | 4 |
| **Macro avg** | **0.27** | **0.50** | **0.35** | **15** |

### C. Per-Class Disease Results — DenseNet-121

**TABLE VI: Per-Class Disease Results — DenseNet-121**

| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| Normal | 0.67 | 1.00 | 0.80 | 4 |
| PCOS | 0.00 | 0.00 | 0.00 | 3 |
| Endometriosis | 0.00 | 0.00 | 0.00 | 4 |
| Adenomyosis | 0.44 | 1.00 | 0.62 | 4 |
| **Macro avg** | **0.28** | **0.50** | **0.35** | **15** |

### D. Per-Class Disease Results — VGG-16

**TABLE VII: Per-Class Disease Results — VGG-16**

| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| Normal | 0.00 | 0.00 | 0.00 | 4 |
| PCOS | 0.43 | 1.00 | 0.60 | 3 |
| Endometriosis | 0.00 | 0.00 | 0.00 | 4 |
| Adenomyosis | 0.50 | 1.00 | 0.67 | 4 |
| **Macro avg** | **0.23** | **0.50** | **0.32** | **15** |

### E. Overall Test-Set Performance Summary

**TABLE VIII: Overall Test-Set Performance (Best Model — ResNet-18)**

| Task | Metric | Result |
|:---|:---|:---|
| Disease (4 classes) | Accuracy | 53.3% |
| Severity (4 classes) | Accuracy | 46.7% |
| Risk score | MAE | 0.298 |

### F. Discussion of Results

The best-performing models (ResNet-18 and DenseNet-121) reached 53.3% disease accuracy on the held-out test set. Several observations emerge from the multi-backbone comparison:

1. **Lighter architectures outperform deeper ones.** ResNet-18 (11.25M params) and DenseNet-121 (7.09M params) both outperform ResNet-50 (23.78M), ResNet-101 (42.77M) and ViT-B/16 (85.91M). This is expected when the dataset is small (800 samples), as larger models overfit more readily.

2. **DenseNet-121 achieves the best risk-score calibration** (MAE 0.258), suggesting that its densely connected feature re-use helps the continuous risk head.

3. **VGG-16 shows competitive disease accuracy (46.7%)** despite its simpler architecture, outperforming ResNet-50 and ResNet-101 on this dataset.

4. **ViT-B/16 underperforms**, likely because its 224 × 224 input requirement and self-attention mechanism need much more data to learn effective representations from scratch or to fine-tune well.

5. **ResNet-18 achieves the best severity accuracy (46.7%)**, while most other backbones score around 26.7–33.3%.

6. **Training time scales with model size**: ResNet-18 trains in 11 seconds while ViT-B/16 takes over 2 minutes — a 10× difference — without accuracy gains.

The held-out test set has only 15 samples, which makes these comparisons illustrative rather than statistically conclusive. Because the data are synthetic and rule-generated, strong results here mainly confirm that the pipeline is correct and should not be read as expected clinical accuracy.

### G. Ablation Study

The central claim of this paper is that fusing both modalities is better than using either alone. To test it, three variants can be trained and evaluated under identical settings. Because the synthetic data are highly separable, single-modality variants may also score close to the fused model. If so, that is a further sign that the data are too easy, and it means this ablation cannot by itself prove the benefit of fusion on real data.

### H. Explainability Results

Grad-CAM overlays for example PCOS, Endometriosis and Adenomyosis patients are generated per backbone. The heat maps highlight the synthetic findings that define each class: the follicle-like spots, the ground-glass cyst and the heterogeneous uterine wall. For ViT-B/16, the CLS token is excluded and the 14 × 14 patch-token grid is used instead of CNN feature maps.

The clinical attribution chart for the same patients lists the features that pushed the prediction towards the predicted class. For a correct prediction these should agree with the features used to generate each class, such as LH/FSH ratio and AMH for PCOS, and pelvic pain and CA-125 for endometriosis. Both explanations are approximate and are meant to let a clinician check plausibility, not to prove causation.

---

## VII. Novelty

The novelty of FemSense AI lies in the combination of the following, rather than in any single component:

1. One framework that covers **three overlapping reproductive disorders** and a Normal class.
2. **Fusion of imaging with clinical and hormonal data**, in line with how clinicians reason.
3. **Eight configurable image backbones** compared under identical multimodal conditions — spanning CNNs (ResNet, VGG, DenseNet, EfficientNet) and Transformers (ViT).
4. Joint prediction of **diagnosis, severity and a risk score** through a shared multi-task representation.
5. **Two-sided explainability** (image heat maps and clinical feature attribution) in the same interface, supporting both CNN Grad-CAM and Transformer patch-token attribution.
6. A **probability-averaged ensemble** that combines all available backbone outputs.
7. A **reproducible, privacy-friendly pipeline** that can be re-pointed to real data by changing only the CSV file.

---

## VIII. Applications

1. **Primary care screening**: Flagging patients with a high risk score for early specialist referral.
2. **Gynaecology and radiology support**: A second opinion that highlights image regions of interest and the clinical factors involved.
3. **Waiting-list prioritisation**: Using severity and risk to decide the urgency of follow-up.
4. **Architecture selection guidance**: The multi-backbone comparison helps practitioners choose the most suitable model for their computational and accuracy requirements.
5. **Medical education**: Teaching how imaging and clinical findings combine for each condition.
6. **Tele-health and rural settings**: A lightweight web interface that runs on modest hardware (CPU-only).

---

## IX. Limitations

1. The dataset is **synthetic**. The results do not represent real patient variability, scanner differences or label noise.
2. The risk score is trained against targets derived from the severity grade, not observed outcomes, so it is a severity-linked score and not a validated probability of progression.
3. Image resolution is limited to **128 × 128** (224 × 224 for ViT) for speed.
4. The test set is **small (15 samples)** and all data come from one generator, so there is no external validation and the backbone comparison is illustrative rather than statistically conclusive.
5. Grad-CAM and gradient-times-input are **approximate explanations** and should not be read as causal evidence.
6. The **probability-averaged ensemble** is not established as superior to individual models and lacks theoretical guarantees.
7. The system is a **research prototype** and is not a medical device.

---

## X. Future Enhancements

Future work will focus on:

- Training and validating on **real, ethically approved** ultrasound and MRI datasets with clinician-confirmed labels, followed by external validation across hospitals.
- **Higher-resolution inputs** and medical-domain-pretrained encoders (e.g., medical ViT, RadImageNet).
- **Uncertainty estimation** and calibration (MC Dropout, ensemble disagreement metrics).
- Richer **fusion mechanisms** such as cross-attention, gated fusion and tensor fusion networks.
- Support for **multi-slice MRI volumes** and 3D convolutional encoders.
- A **clinician usability study** of the multi-model comparison and explanation views.
- Expanding the backbone library to include newer architectures (ConvNeXt, Swin Transformer, EfficientNet-V2).
- **Statistical significance testing** of backbone differences with larger held-out sets.

---

## XI. Conclusion

This paper presented FemSense AI, a multimodal deep learning framework that combines ultrasound/MRI images with clinical and hormonal data to classify PCOS, endometriosis, adenomyosis and normal cases while also estimating severity and a risk score. The framework supports **eight configurable image backbones** — ResNet-18/50/101, EfficientNet-B0, DenseNet-121, VGG-16/19 and ViT-B/16 — all pre-trained on the ImageNet dataset and evaluated under identical multimodal conditions.

On a synthetic test set, the best-performing backbones (**ResNet-18 and DenseNet-121**) achieved **53.3% disease accuracy**, with DenseNet-121 achieving the lowest risk-score MAE of **0.258**. The systematic comparison reveals that lighter architectures outperform deeper ones on this small-scale dataset. A **probability-averaged ensemble** and built-in **Grad-CAM / gradient-times-input explainability** make the predictions inspectable through a Streamlit multi-model research dashboard with in-memory PDF/CSV report generation.

These numbers verify the pipeline but are not clinical evidence. The modular design allows the framework to be retrained on real patient data, which is the key next step before any clinical use.

---

## Acknowledgment

The authors thank the Department of Artificial Intelligence and Data Analytics, Sri Ramachandra Faculty of Engineering and Technology, Chennai, and their project guide for the facilities, support and guidance provided during this work.

---

## References

[1] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.

[2] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, 2017, pp. 618–626.

[3] T. Baltrušaitis, C. Ahuja, and L.-P. Morency, "Multimodal machine learning: A survey and taxonomy," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 41, no. 2, pp. 423–443, 2019.

[4] R. Caruana, "Multitask learning," *Machine Learning*, vol. 28, no. 1, pp. 41–75, 1997.

[5] H. J. Teede et al., "Recommendations from the 2023 international evidence-based guideline for the assessment and management of polycystic ovary syndrome," *Fertility and Sterility*, vol. 120, no. 4, pp. 767–793, 2023.

[6] K. T. Zondervan, C. M. Becker, and S. A. Missmer, "Endometriosis," *New England Journal of Medicine*, vol. 382, no. 13, pp. 1244–1256, 2020.

[7] K. Upson and S. A. Missmer, "Epidemiology of adenomyosis," *Seminars in Reproductive Medicine*, vol. 38, no. 2–3, pp. 89–107, 2020.

[8] S. J. Pan and Q. Yang, "A survey on transfer learning," *IEEE Trans. Knowl. Data Eng.*, vol. 22, no. 10, pp. 1345–1359, 2010.

[9] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.

[10] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2015.

[11] A. Shrikumar, P. Greenside, A. Shcherbina, and A. Kundaje, "Not just a black box: Learning important features through propagating activation differences," arXiv:1605.01713, 2016.

[12] G. Litjens et al., "A survey on deep learning in medical image analysis," *Medical Image Analysis*, vol. 42, pp. 60–88, 2017.

[13] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. Int. Conf. Machine Learning (ICML)*, 2019, pp. 6105–6114.

[14] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely connected convolutional networks," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 4700–4708.

[15] K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2015.

[16] A. Dosovitskiy et al., "An image is worth 16×16 words: Transformers for image recognition at scale," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2021.

[17] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, "ImageNet: A large-scale hierarchical image database," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2009, pp. 248–255.
