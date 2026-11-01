"""
Synthetic data generator.

Real MRI/ultrasound datasets of PCOS / Endometriosis / Adenomyosis are
patient-sensitive and need ethics approval, so this script creates a
SYNTHETIC, ultrasound-like image set plus correlated clinical records so the
whole pipeline runs end-to-end out of the box.

To use REAL data: keep the same CSV columns (config.CLINICAL_FEATURES plus
image_path, disease, severity) and put images in data/images/.
"""
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from src.config import IMG_DIR, CSV_PATH, DISEASES, SEED

SYNTHETIC_IMAGE_SIZE = 128


def _grid(n):
    y, x = np.mgrid[0:n, 0:n]
    return x.astype(np.float32), y.astype(np.float32)


def _blob(x, y, cx, cy, rx, ry, angle=0.0):
    c, s = np.cos(angle), np.sin(angle)
    xr, yr = (x - cx) * c + (y - cy) * s, -(x - cx) * s + (y - cy) * c
    return ((xr / rx) ** 2 + (yr / ry) ** 2) <= 1.0


def make_image(disease, severity, rng, n=SYNTHETIC_IMAGE_SIZE):
    """Return a speckled grayscale ultrasound-like image in [0,1]."""
    x, y = _grid(n)
    img = rng.normal(0.35, 0.08, (n, n)).astype(np.float32)
    cx, cy = n / 2 + rng.uniform(-6, 6), n / 2 + rng.uniform(-6, 6)
    organ = _blob(x, y, cx, cy, n * 0.36, n * 0.28, rng.uniform(-0.4, 0.4))
    img[organ] += 0.18

    if disease == "PCOS":  # many small anechoic follicles ("string of pearls")
        k = 6 + 4 * severity + int(rng.integers(0, 3))
        for _ in range(k):
            ang = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0.15, 0.3) * n
            fx, fy = cx + r * np.cos(ang), cy + r * 0.8 * np.sin(ang)
            rad = rng.uniform(2.0, 4.0)
            img[_blob(x, y, fx, fy, rad, rad)] -= 0.30
    elif disease == "Endometriosis":  # ground-glass endometrioma (cyst)
        rad = 9 + 6 * severity + rng.uniform(-2, 2)
        fx, fy = cx + rng.uniform(-10, 10), cy + rng.uniform(-8, 8)
        m = _blob(x, y, fx, fy, rad, rad * 0.85)
        img[m] = 0.45 + rng.normal(0, 0.03, int(m.sum()))
        ring = _blob(x, y, fx, fy, rad + 2, (rad + 2) * 0.85) & ~m
        img[ring] += 0.25
    elif disease == "Adenomyosis":  # heterogeneous myometrium + streaks
        organ2 = _blob(x, y, cx, cy, n * (0.38 + 0.03 * severity), n * 0.30)
        img[organ2] += rng.normal(0.05, 0.10 + 0.04 * severity, int(organ2.sum()))
        for _ in range(3 + 2 * severity):
            sx, sy = rng.uniform(cx - 25, cx + 25), rng.uniform(cy - 18, cy + 18)
            img[_blob(x, y, sx, sy, rng.uniform(1, 2.5), rng.uniform(6, 12),
                      rng.uniform(0, np.pi))] -= 0.25

    img += rng.normal(0, 0.06, img.shape)  # speckle noise
    return np.clip(img, 0, 1)


def make_clinical(disease, severity, rng):
    s = severity
    r = dict(
        age=rng.integers(18, 45), bmi=rng.normal(23, 3), cycle_length_days=rng.normal(28, 2),
        pelvic_pain_score=rng.uniform(0, 2), dysmenorrhea_score=rng.uniform(0, 3),
        heavy_bleeding=int(rng.random() < 0.1), hirsutism_score=rng.uniform(0, 4),
        acne=int(rng.random() < 0.2), infertility=int(rng.random() < 0.08),
        family_history=int(rng.random() < 0.1), amh=rng.normal(3, 1),
        lh_fsh_ratio=rng.normal(1.0, 0.3), testosterone=rng.normal(35, 10), ca125=rng.normal(15, 6),
    )
    if disease == "PCOS":
        r.update(bmi=rng.normal(27 + s, 4), cycle_length_days=rng.normal(40 + 6 * s, 6),
                 hirsutism_score=rng.uniform(8, 14 + 2 * s), acne=int(rng.random() < 0.7),
                 amh=rng.normal(6 + 1.5 * s, 1.5), lh_fsh_ratio=rng.normal(2.3 + 0.3 * s, 0.4),
                 testosterone=rng.normal(65 + 8 * s, 12), infertility=int(rng.random() < 0.5))
    elif disease == "Endometriosis":
        r.update(pelvic_pain_score=rng.uniform(5 + s, 10), dysmenorrhea_score=rng.uniform(6 + s * 0.8, 10),
                 ca125=rng.normal(40 + 20 * s, 12), infertility=int(rng.random() < 0.45),
                 family_history=int(rng.random() < 0.3), amh=rng.normal(2.0, 0.8))
    elif disease == "Adenomyosis":
        r.update(heavy_bleeding=int(rng.random() < 0.9), dysmenorrhea_score=rng.uniform(5 + s, 10),
                 pelvic_pain_score=rng.uniform(4 + s * 0.7, 9), age=rng.integers(30, 50),
                 ca125=rng.normal(30 + 12 * s, 10))
    r["age"] = int(r["age"])
    for k in ("bmi", "cycle_length_days", "hirsutism_score", "amh", "lh_fsh_ratio", "testosterone",
              "ca125", "pelvic_pain_score", "dysmenorrhea_score"):
        r[k] = round(float(max(r[k], 0.1)), 2)
    return r


def main(n_samples=800):
    rng = np.random.default_rng(SEED)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_samples):
        d = int(rng.integers(0, len(DISEASES)))
        disease = DISEASES[d]
        sev = 0 if disease == "Normal" else int(rng.integers(1, 4))
        img = make_image(disease, sev, rng)
        path = IMG_DIR / f"img_{i:04d}.png"
        Image.fromarray((img * 255).astype(np.uint8)).save(path)
        row = make_clinical(disease, sev, rng)
        row.update(image_path=f"images/{path.name}", disease=d, severity=sev)
        rows.append(row)
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
    print(f"Generated {n_samples} samples -> {CSV_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    main(ap.parse_args().n)
