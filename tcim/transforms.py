from typing import Literal

import cv2
import numpy as np
from scipy import ndimage


def translation(img, x, y):
    h, w = img.shape
    img_new = np.zeros((h + y * 2, w + x * 2), dtype=np.uint8)
    img_new[y:y + h, x:x + w] = img
    return img_new


def rotation(img, degree):
    return ndimage.rotate(img, degree)


def flip(img, flip_type: Literal['vertical', 'horizontal']):
    return cv2.flip(img, 0 if flip_type == 'vertical' else 1)


def scale(img, scale_x, scale_y):
    return cv2.resize(img, None, fx=scale_x, fy=scale_y)


def noise(img, intensity):
    h, w = img.shape
    mask = img > 0
    n = np.clip(np.random.randn(h, w)*256*intensity, a_min=0, a_max=255).astype(np.uint8) * mask
    return cv2.add(img, n)
