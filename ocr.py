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

if __name__ == "__main__":
    # text = ocr_image(SAMPLES_DIR / "ua_screen.png")
    # print(text[:300])

    print(round(mean_confidence(SAMPLES_DIR / "en_screen.png"), 1))
    print(round(mean_confidence(SAMPLES_DIR / "photo.jpg"), 1))
