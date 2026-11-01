# FemSense AI
**Early Detection of Women's Reproductive Disorders Using Multimodal Deep Learning**
Team 8 - R. Bavadhaarini (E0324001), Lishalini M (E0324002)

Detects **PCOS, Endometriosis, Adenomyosis** (or Normal), and predicts **severity** and a **risk score**
by fusing ultrasound/MRI images with clinical data. Includes Grad-CAM + feature-attribution explainability.

## Configurable image backbones

The image encoder is selectable while the clinical branch, feature set, fusion,
three prediction heads, and multi-task loss remain unchanged.

| Backbone | Input size | Image feature width |
|---|---:|---:|
| `resnet18` | 128 | 512 |
| `resnet50` | 128 | 2048 |
| `resnet101` | 128 | 2048 |
| `efficientnet_b0` | 128 | 1280 |
| `densenet121` | 128 | 1024 |
| `vgg16` | 128 | 512 |
| `vgg19` | 128 | 512 |
| `vit_b_16` | 224 | 768 |

Checkpoints are stored separately as `models/femsense_<backbone>.pt`.
The scaler and seeded test split are shared at `models/scaler.pkl` and
`models/test_split.csv`.

## Architecture
```
Image (US/MRI) -> configurable image backbone -> image features ──┐
                                                               ├─ concat -> Fusion MLP(128) -> Disease head (4)
Clinical data (14 features) -> Dense NN -> 32-d ───────────────┘                           -> Severity head (4)
                                                                                           -> Risk head (0-1)
Explainability: Grad-CAM (image) + gradient x input (clinical)
```

## Run in VS Code (Windows / Mac / Linux)
1. Install Python 3.9-3.12. Open this folder in VS Code (`File > Open Folder`).
2. Open a terminal (`Ctrl+``) and create an environment:
   ```
   python -m venv venv
   venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
   pip install -r requirements.txt
   ```
   Select the interpreter: `Ctrl+Shift+P` -> *Python: Select Interpreter* -> `venv`.
3. Run the pipeline (always from the project root):
   ```
   python -m src.generate_data     # creates 800 synthetic samples in data/
   python -m src.train --backbone resnet18 --epochs 12
   python -m src.evaluate --backbone resnet18
   python -m src.predict --index 5 --backbone resnet18
   python -m src.compare --backbones all --epochs 12
   streamlit run app.py            # web demo at http://localhost:8501
   ```
   You can also use the Run & Debug panel (configs are in `.vscode/launch.json`).

Add `--pretrained` to use torchvision ImageNet weights (requires internet for the
first download). The default learning rate is `1e-4` with pretrained weights and
`1e-3` from scratch. Training runs on CPU; ResNet101 and ViT default to batch
size 16. Override with `--batch_size`.

`src.compare` trains and evaluates each selected backbone, records parameter
counts and training time, and writes `outputs/comparison.csv` and
`outputs/comparison.png`. A failed run is logged in
`outputs/comparison_failures.log` without stopping later backbones. For a
smaller smoke run use `--epochs 1 --max_samples 100`. Per-backbone histories,
curves, metrics, and confusion matrices are written under `outputs/<backbone>/`.

The Streamlit dashboard evaluates available checkpoints on the shared
`models/test_split.csv`, highlights the highest disease-accuracy model, and lets
you upload an image plus the 14 clinical fields (or a CSV containing those
fields) to run all checkpoints sequentially. It shows individual results,
a probability-averaged ensemble (also measured on the held-out split), comparison
metrics and plots, model explanations, and in-memory PDF/CSV downloads.
The bundled generator produces synthetic data; displayed evaluation metrics
are not clinical evidence. Uploaded patient data and reports are not written to
disk. Run it with `streamlit run app.py`.

## IMPORTANT: data
The bundled generator creates **synthetic** ultrasound-like images and correlated clinical records so the
project runs without patient data. Accuracy on it is NOT clinical evidence. For your final report, plug in
real datasets (e.g. public PCOS ultrasound sets on Kaggle, UTERUS/endometriosis MRI sets) by writing a
`data/clinical_data.csv` with columns: the 14 clinical features in `src/config.py`, plus
`image_path` (relative to `data/`), `disease` (0-3), `severity` (0-3). Use the
`--pretrained` training flag for ImageNet transfer learning (needs internet on
the first run).

## Project structure
```
app.py              Streamlit demo
src/config.py       paths, labels, hyper-parameters
src/generate_data.py synthetic data generator
src/dataset.py      Dataset + transforms
src/model.py        multimodal network
src/train.py        training loop (multi-task loss)
src/evaluate.py     test metrics, confusion matrices
src/compare.py      train/evaluate backbone comparison
src/inference.py    predict + Grad-CAM + clinical attribution
src/predict.py      CLI demo
```
*Research prototype / decision support only - not a medical device.*
