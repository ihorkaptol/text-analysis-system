import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps


def to_grayscale(image):
    return image.convert("L")


def binarize(image, threshold=150):
    gray = to_grayscale(image)
    return gray.point(lambda x: 0 if x < threshold else 255)


def autocontrast(image, cutoff=1):
    gray = to_grayscale(image)
    return ImageOps.autocontrast(gray, cutoff=cutoff)


def otsu_threshold_value(image):
    pixels = np.array(to_grayscale(image))
    t, _ = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(t)


def otsu_binarize(image):
    pixels = np.array(to_grayscale(image))
    _, binary = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)


def upscale(image, factor=2):
    w, h = image.size
    return image.resize((int(w * factor), int(h * factor)), Image.Resampling.LANCZOS)


def median_denoise(image, size=3):
    return image.filter(ImageFilter.MedianFilter(size=size))


STEPS = {
    "gray": to_grayscale,
    "autocontrast": autocontrast,
    "upscale": upscale,
    "median": median_denoise,
    "otsu": otsu_binarize
}


def preprocess(image, steps):
    result = image
    for name in steps:
        if name not in STEPS:
            raise ValueError(f"Unknown step: {name!r}. Available: {list(STEPS)}")
        result = STEPS[name](result)
    return result


if __name__ == "__main__":
    from ocr import SAMPLES_DIR, mean_confidence

    photo = Image.open(SAMPLES_DIR / "photo.jpg")
    print("otsu threshold:", otsu_threshold_value(photo))
    print("conf original:", round(mean_confidence(photo), 1))
    print("conf otsu:", round(mean_confidence(otsu_binarize(photo)), 1))
    print(photo.size, upscale(photo).size)

    out = preprocess(photo, ["gray", "upscale", "otsu"])
    print(out.mode, out.size)
    print(sorted(set(np.array(out).flatten().tolist())))
