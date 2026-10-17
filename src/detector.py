import json
import os
import cv2
import numpy as np


def choose_detection_method(image_path: str) -> str:
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Kanał S (Saturation / Nasycenie)
    saturation = hsv[:, :, 1]

    # Liczymy, ile pikseli ma wyraźny, nasycony kolor (np. różowe/zielone kropki)
    color_pixels_count = np.sum(saturation > 60)

    # Jeśli jest chociaż kilkadziesiąt kolorowych pikseli (rozmiar paru kropek),
    # używamy HSV. Jeśli nie – przełączamy się na odcienie szarości (Gray).
    if color_pixels_count > 50:
        return "hsv"
    else:
        return "gray_shape"
