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

if __name__ == "__main__":
    dull = Image.linear_gradient("L").point(lambda p: 90 + p * 100 // 255)
    print(dull.getextrema(), autocontrast(dull).getextrema())
