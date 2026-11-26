import csv
from pathlib import Path

from PIL import Image

from ocr import SAMPLES_DIR, mean_confidence, ocr_image
from preprocessing import preprocess

RESULTS_DIR = Path(__file__).parent / "results"
PROCESSED_DIR = SAMPLES_DIR.parent / "processed"
VARIANTS = {
    "original": [],
    "otsu": ["gray", "otsu"],
    "contrast+otsu": ["gray", "autocontrast", "otsu"],
    "median+otsu": ["gray", "median", "otsu"],
    "x2+otsu": ["gray", "upscale", "otsu"],
}

if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)
    images = sorted(
        p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
    )
    rows = []
    print(f"{'file':<16}{'variant':<17}{'words':>6}{'conf':>7}")
    for path in images:
        original = Image.open(path)
        for variant, steps in VARIANTS.items():
            image = preprocess(original, steps)
            image.save(PROCESSED_DIR / f"{path.stem}_{variant}.png")
            text = ocr_image(image)
            words = len(text.split())
            conf = mean_confidence(image)
            rows.append({"file": path.name, "variant": variant, "words": words, "conf": round(conf, 1)})
            print(f"{path.name:<16}{variant:<17}{words:>6}{conf:>7.1f}")

    out_path = RESULTS_DIR / "preprocessing_comparison.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "variant", "words", "conf"])
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", out_path)