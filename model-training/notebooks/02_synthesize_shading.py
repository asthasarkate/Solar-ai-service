"""
Solara AI — Synthetic Shading Generation
Generates ~350 synthetic shadow overlay images from clean panel photos.

DISCLOSURE: The Shading class contains synthetically generated images
(simulated shadow overlays on clean panel photos), as no source dataset
provided real shading examples. This should be disclosed in the project
report and ideally supplemented with real images if available before
final submission.
"""
import os
import sys
import random
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

MODEL_TRAINING = Path(__file__).resolve().parent.parent
RAW_DIR = MODEL_TRAINING / "dataset_raw"
OUTPUT_DIR = RAW_DIR / "synthetic_shading" / "shading"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif'}
TARGET_COUNT = 350


# ── Collect clean source images ──────────────────────────────────
def collect_clean_images():
    clean_paths = []

    s2 = RAW_DIR / "source_2_dust" / "Clean"
    if s2.exists():
        paths = [p for p in s2.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        clean_paths.extend(paths)
        print(f"  source_2_dust/Clean: {len(paths)}")

    for split in ['train', 'val', 'test']:
        for src in ['source_1_afroz', 'source_3_pv_defect']:
            d = RAW_DIR / src / split / "Clean"
            if d.exists():
                paths = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
                clean_paths.extend(paths)
                print(f"  {src}/{split}/Clean: {len(paths)}")

    top = RAW_DIR / "source_1_afroz" / "Clean"
    if top.exists():
        paths = [p for p in top.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        clean_paths.extend(paths)
        print(f"  source_1_afroz/Clean (top): {len(paths)}")

    return clean_paths


# ── Shadow generators ────────────────────────────────────────────
def create_soft_mask(size, polygon_points, blur_radius=25):
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon_points, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return mask


def apply_shadow_multiply(img_arr, mask_arr, darkness):
    mask_norm = mask_arr.astype(np.float32) / 255.0
    result = img_arr.astype(np.float32)
    for c in range(3):
        result[:, :, c] *= (1.0 - mask_norm * darkness)
    return np.clip(result, 0, 255).astype(np.uint8)


def polygon_shadow(w, h):
    n = random.randint(4, 8)
    cx = random.randint(w // 4, 3 * w // 4)
    cy = random.randint(h // 4, 3 * h // 4)
    diag = np.sqrt(w**2 + h**2)
    radius = random.uniform(0.15, 0.40) * diag
    angles = np.sort(np.random.uniform(0, 2 * np.pi, n))
    pts = []
    for a in angles:
        r = radius * random.uniform(0.5, 1.5)
        pts.append((int(np.clip(cx + r * np.cos(a), 0, w)),
                     int(np.clip(cy + r * np.sin(a), 0, h))))
    return pts


def gradient_shadow(w, h):
    angle = random.uniform(0, 360)
    Y, X = np.mgrid[0:h, 0:w]
    rad = np.radians(angle)
    proj = X * np.cos(rad) + Y * np.sin(rad)
    g = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
    g = np.power(g, random.uniform(0.5, 2.0))
    return Image.fromarray((g * 255).astype(np.uint8), mode='L')


def radial_shadow(w, h):
    side = random.choice(['tl', 'tr', 'bl', 'br', 'top', 'bottom', 'left', 'right'])
    ox = {'tl': 0, 'bl': 0, 'tr': w, 'br': w, 'top': random.randint(0, w),
           'bottom': random.randint(0, w), 'left': 0, 'right': w}[side]
    oy = {'tl': 0, 'tr': 0, 'bl': h, 'br': h, 'top': 0, 'bottom': h,
           'left': random.randint(0, h), 'right': random.randint(0, h)}[side]
    Y, X = np.mgrid[0:h, 0:w]
    dist = np.sqrt((X - ox)**2 + (Y - oy)**2)
    g = 1.0 - dist / np.sqrt(w**2 + h**2)
    g = np.clip(g, 0, 1)
    return Image.fromarray((g * 255).astype(np.uint8), mode='L')


# ── Main synthesis ───────────────────────────────────────────────
def generate_one(img_path, out_path):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    img_arr = np.array(img)
    darkness = random.uniform(0.20, 0.60)
    blur_r = random.randint(15, 40)

    method = random.choice(['polygon', 'gradient', 'radial'])
    if method == 'polygon':
        mask = create_soft_mask((w, h), polygon_shadow(w, h), blur_r)
    elif method == 'gradient':
        mask = gradient_shadow(w, h).filter(ImageFilter.GaussianBlur(radius=blur_r))
    else:
        mask = radial_shadow(w, h).filter(ImageFilter.GaussianBlur(radius=blur_r))

    mask_arr = np.array(mask)
    result = apply_shadow_multiply(img_arr, mask_arr, darkness)
    Image.fromarray(result).save(out_path, quality=92)
    return method, darkness


def main():
    print("=" * 60)
    print("SOLARA AI — SYNTHETIC SHADING GENERATION")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"Target: {TARGET_COUNT} images\n")

    print("Collecting clean source images:")
    clean = collect_clean_images()
    print(f"\nTotal clean images: {len(clean)}")
    print(f"Generating {TARGET_COUNT} synthetic shading images...\n")

    queue = list(clean)
    random.shuffle(queue)
    idx = 0
    generated = 0
    methods = {'polygon': 0, 'gradient': 0, 'radial': 0}

    while generated < TARGET_COUNT:
        if idx >= len(queue):
            idx = 0
            random.shuffle(queue)

        out_name = f"synthetic_shading_{generated + 1:04d}.jpg"
        out_path = OUTPUT_DIR / out_name

        try:
            method, darkness = generate_one(queue[idx], out_path)
            methods[method] += 1
            generated += 1
            if generated % 50 == 0:
                print(f"  [{generated}/{TARGET_COUNT}] done")
        except Exception as e:
            print(f"  Skip: {e}")

        idx += 1

    print(f"\nDone! Generated {generated} images")
    print(f"Methods used: {methods}")
    print(f"Saved to: {OUTPUT_DIR}")

    # ── Visual validation ────────────────────────────────────────
    samples = random.sample(list(OUTPUT_DIR.glob('*.jpg')), min(12, generated))
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Synthetic Shading Samples — Visual Validation', fontsize=16, fontweight='bold')
    for i, ax in enumerate(axes.flat):
        if i < len(samples):
            ax.imshow(Image.open(samples[i]))
            ax.set_title(samples[i].name, fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(MODEL_TRAINING / 'notebooks' / 'synthetic_shading_validation.png', dpi=150)
    plt.close()
    print(f"Validation grid saved to notebooks/synthetic_shading_validation.png")


if __name__ == '__main__':
    main()
