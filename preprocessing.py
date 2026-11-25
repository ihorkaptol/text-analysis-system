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


if __name__ == "__main__":
    from ocr import SAMPLES_DIR, mean_confidence

    photo = Image.open(SAMPLES_DIR / "photo.jpg")
    print("otsu threshold:", otsu_threshold_value(photo))
    print("conf original:", round(mean_confidence(photo), 1))
    print("conf otsu:", round(mean_confidence(otsu_binarize(photo)), 1))
