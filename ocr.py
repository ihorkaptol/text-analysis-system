from pathlib import Path

import pytesseract
from pytesseract import Output
from PIL import Image

SAMPLES_DIR = Path(__file__).parent / "data" / "samples"
LANG = "ukr+eng"


def ocr_image(path, lang=LANG, psm=3):
    img = Image.open(path)
    return pytesseract.image_to_string(img, lang=lang, config=f"--psm {psm}")

def mean_confidence(path, lang=LANG, psm=3):
    img = Image.open(path)
    data = pytesseract.image_to_data(img, lang=lang, config=f"--psm {psm}", output_type=Output.DICT)
    confs = [c for c, word in zip(data["conf"], data["text"]) if c >= 0 and word.strip()]
    if not confs:
        return 0.0
    return sum(confs) / len(confs)

def to_greyscale(image):
    return image.convert("L")

def binarize(image, threshold=150):
    gray = to_greyscale(image)
    return gray.point(lambda x: 0 if x < threshold else 255)

if __name__ == "__main__":

    images = sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"{'file':<16}{'psm':>4}{'words':>7}{'conf':>8}")
    for path in images:
        for psm in (3, 6):
            text = ocr_image(path, lang=LANG, psm=psm)
            words = len(text.split())
            conf = mean_confidence(path, lang=LANG, psm=psm)
            print(f"{path.name:<16}{psm:>4}{words:>7}{conf:>8.1f}")

    text = "Parallel computing is\na type"
    print(text.split())

    photo = SAMPLES_DIR / "photo.jpg"
    out_dir = SAMPLES_DIR.parent / "processed"
    out_dir.mkdir(exist_ok=True)
    binarize(Image.open(photo), 150).save(out_dir / "photo_bin_150.png")
    print("original", round(mean_confidence(photo), 1))
    print("bin 150", round(mean_confidence(out_dir / "photo_bin_150.png"), 1))