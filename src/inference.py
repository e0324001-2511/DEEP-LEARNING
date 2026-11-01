"""Shared inference + explainability helpers (used by CLI and web app)."""
import joblib
import numpy as np
import torch
import torch.nn.functional as F

from src import config as C
from src.dataset import preprocess_image
from src.model import FemSenseNet


def load_model(backbone=None, device="cpu"):
    """Load a backbone checkpoint and its shared clinical scaler."""
    backbone = backbone or C.DEFAULT_BACKBONE
    path = C.model_path(backbone)
    if not path.exists() and backbone == C.DEFAULT_BACKBONE and C.MODEL_PATH.exists():
        path = C.MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model for {backbone} not found. Run: python -m src.train --backbone {backbone}"
        )
    ck = torch.load(path, map_location=device, weights_only=False)
    checkpoint_backbone = ck.get("backbone", backbone)
    if checkpoint_backbone != backbone:
        raise ValueError(
            f"Checkpoint at {path} is for {checkpoint_backbone}, not {backbone}."
        )
    model = FemSenseNet(ck["n_clinical"], backbone=checkpoint_backbone, pretrained=False)
    model.img_size = ck.get("img_size", C.get_img_size(checkpoint_backbone))
    state_dict = ck["state_dict"]
    if "backbone.conv1.weight" not in state_dict and "cnn.conv1.weight" in state_dict:
        state_dict = {
            ("backbone." + key[4:] if key.startswith("cnn.") else key): value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict)
    model.to(device).eval()
    scaler = joblib.load(C.MODEL_DIR / "scaler.pkl")
    return model, scaler


def _clin_tensor(scaler, clinical: dict, device):
    x = np.array([[clinical[k] for k in C.CLINICAL_FEATURES]], dtype=np.float32)
    return torch.from_numpy(scaler.transform(x).astype(np.float32)).to(device)


def risk_level(r):
    return "Low" if r < 0.25 else "Moderate" if r < 0.55 else "High" if r < 0.8 else "Very high"


def recommendation(disease, severity, risk):
    base = {
        "Normal": "No strong indication of a reproductive disorder. Routine check-ups are advised.",
        "PCOS": "Suggest endocrine evaluation, lifestyle/weight management, and cycle regulation review.",
        "Endometriosis": "Suggest gynaecological review, pain management plan and consideration of laparoscopy/MRI follow-up.",
        "Adenomyosis": "Suggest gynaecological review, transvaginal ultrasound/MRI confirmation and bleeding management.",
    }[disease]
    urgency = {"None": "", "Mild": " Follow-up in 6-12 months.", "Moderate": " Specialist consult within 1-3 months.",
               "Severe": " Prompt specialist consultation recommended."}[severity]
    return base + urgency + " (Decision-support only - not a medical diagnosis.)"


def predict(model, scaler, pil_img, clinical: dict, device="cpu"):
    img = preprocess_image(pil_img, model.img_size).unsqueeze(0).to(device)
    clin = _clin_tensor(scaler, clinical, device)
    with torch.no_grad():
        ld, ls, lr = model(img, clin)
    pd_, ps = F.softmax(ld, 1)[0].cpu().numpy(), F.softmax(ls, 1)[0].cpu().numpy()
    risk = float(torch.sigmoid(lr)[0])
    d, s = C.DISEASES[int(pd_.argmax())], C.SEVERITIES[int(ps.argmax())]
    return {"disease": d, "disease_probs": dict(zip(C.DISEASES, pd_.tolist())),
            "severity": s, "severity_probs": dict(zip(C.SEVERITIES, ps.tolist())),
            "risk": risk, "risk_level": risk_level(risk), "advice": recommendation(d, s, risk)}


def grad_cam(model, pil_img, clinical, scaler, class_idx=None, device="cpu"):
    """Return a Grad-CAM heat-map (H,W in [0,1]) for a disease class."""
    img = preprocess_image(pil_img, model.img_size).unsqueeze(0).to(device)
    clin = _clin_tensor(scaler, clinical, device)
    store = {}

    def fwd(_, __, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    h = model.get_cam_target().register_forward_hook(fwd)
    model.eval()
    try:
        with torch.enable_grad():
            ld, _, _ = model(img, clin)
            idx = int(ld.argmax(1)) if class_idx is None else class_idx
            model.zero_grad()
            ld[0, idx].backward()
    finally:
        h.remove()

    activation, gradient = store["a"], store["g"]
    if activation.ndim == 3:
        # ViT activations are (batch, CLS + patch tokens, channels).
        activation = activation[:, 1:, :].transpose(1, 2).reshape(
            activation.shape[0], activation.shape[2], 14, 14
        )
        gradient = gradient[:, 1:, :].transpose(1, 2).reshape(
            gradient.shape[0], gradient.shape[2], 14, 14
        )
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activation).sum(1, keepdim=True))
    cam = F.interpolate(
        cam, size=(model.img_size, model.img_size),
        mode="bilinear", align_corners=False,
    )[0, 0]
    cam = cam.detach().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def clinical_importance(model, pil_img, clinical, scaler, class_idx=None, device="cpu"):
    """Gradient x input attribution for each clinical feature (signed)."""
    img = preprocess_image(pil_img, model.img_size).unsqueeze(0).to(device)
    clin = _clin_tensor(scaler, clinical, device).requires_grad_(True)
    model.eval()
    with torch.enable_grad():
        ld, _, _ = model(img, clin)
        idx = int(ld.argmax(1)) if class_idx is None else class_idx
        model.zero_grad()
        ld[0, idx].backward()
    attr = (clin.grad * clin).detach().cpu().numpy()[0]
    return dict(zip(C.CLINICAL_FEATURES, attr.tolist()))
