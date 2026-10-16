import json
import math
import os
import shutil
import sys
import cv2
import numpy as np

DEFAULT_RAW_DIR = os.path.join("data", "raw_data")
RAW_DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RAW_DIR

IMAGES_DIR = os.path.join(RAW_DATA_DIR, "images", "png")
ANNOTATIONS_JSON_PATH = os.path.join(
    RAW_DATA_DIR, "annotations", "annotations.json"
)

TARGET_DIR = os.path.join("data", "sample_images")
os.makedirs(TARGET_DIR, exist_ok=True)

if not os.path.exists(ANNOTATIONS_JSON_PATH):
    print(f"nie znaleziono pliku adnotacji w ścieżce: {ANNOTATIONS_JSON_PATH}")
    sys.exit(1)

with open(ANNOTATIONS_JSON_PATH, "r", encoding="utf-8") as f:
    all_annotations = json.load(f)

color_samples = []
monochromatic_samples = []


def get_dynamic_plot_roi(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=int(min(h, w) * 0.3),
        maxLineGap=5,
    )

    x_axis_left = int(w * 0.12)
    y_axis_bottom = int(h * 0.82)

    if lines is not None:
        vertical_x = []
        horizontal_y = []

        for line in lines:
            x1, y1, x2, y2 = line.ravel()

            if abs(x1 - x2) < 3 and abs(y1 - y2) > h * 0.3:
                if x1 < w * 0.4:
                    vertical_x.append(x1)

            elif abs(y1 - y2) < 3 and abs(x1 - x2) > w * 0.3:
                if y1 > h * 0.5:
                    horizontal_y.append(y1)

        if vertical_x:
            x_axis_left = max(vertical_x) + 2
        if horizontal_y:
            y_axis_bottom = min(horizontal_y) - 2

    y_title_top = int(h * 0.08)
    x_right_limit = int(w * 0.98)

    return img[y_title_top:y_axis_bottom, x_axis_left:x_right_limit]


def has_detectable_dots(img, is_color=True):
    roi = get_dynamic_plot_roi(img)

    if is_color:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv, np.array([0, 30, 20]), np.array([179, 255, 245])
        )
    else:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    valid_dots = []

    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)

        if 8 <= area <= 400 and perimeter > 0:
            circularity = (4 * math.pi * area) / (perimeter**2)
            if circularity >= 0.60:
                valid_dots.append(c)

    return len(valid_dots) >= 1


for item in all_annotations:
    is_dot_line = item.get("type") == "dot_line"
    has_single_series = len(item.get("models", [])) == 1

    has_json_data = False
    if has_single_series:
        model = item["models"][0]
        if "y" in model and len(model["y"]) > 0:
            has_json_data = True

    if is_dot_line and has_single_series and has_json_data:
        img_idx = item["image_index"]
        img_path = os.path.join(IMAGES_DIR, f"{img_idx}.png")

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                color_pixels = np.sum(hsv[:, :, 1] > 50)

                if color_pixels > 100 and len(color_samples) < 24:
                    if has_detectable_dots(img, is_color=True):
                        color_samples.append(item)

                elif color_pixels <= 100 and len(monochromatic_samples) < 6:
                    if has_detectable_dots(img, is_color=False):
                        monochromatic_samples.append(item)

        if len(color_samples) == 24 and len(monochromatic_samples) == 6:
            break

print(
    f"Wyselekcjonowano: {len(color_samples)} wykresów kolorowych i"
    f" {len(monochromatic_samples)} monochromatyczne."
)

dev_set = color_samples[:7] + monochromatic_samples[:3]
test_set = color_samples[7:24] + monochromatic_samples[3:6]
selected_samples = dev_set + test_set

dataset_metadata = {
    "dataset_name": "PlotQA",
    "original_split": "validation",
    "selection_criteria": (
        "30 wykresów typu 'dot_line'"
        " (24 kolorowe + 6 monochromatycznych). Stratyfikacja: DEV (7 kolor / 3 monochromatyczne),"
        " TEST (17 kolor / 3 monochromatyczne)."
    ),
    "dev_ids": [item["image_index"] for item in dev_set],
    "test_ids": [item["image_index"] for item in test_set],
    "samples": selected_samples,
}

metadata_path = os.path.join("data", "sample_annotations.json")
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(dataset_metadata, f, indent=4, ensure_ascii=False)

copied_count = 0
for item in selected_samples:
    img_idx = item["image_index"]
    img_name = f"{img_idx}.png"

    src_path = os.path.join(IMAGES_DIR, img_name)
    dst_path = os.path.join(TARGET_DIR, img_name)

    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        copied_count += 1

print(f"Skopiowano {copied_count}/30 plików do folderu {TARGET_DIR}.")
print(f"Identyfikatory DEV (10):  {[item['image_index'] for item in dev_set]}")
print(f"Identyfikatory TEST (20): {[item['image_index'] for item in test_set]}")