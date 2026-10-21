import csv
import json
import numpy as np

def extract_calibration_points(calibration_config):
    y_rule = calibration_config["y_axis"]["rule"]["bbox"]
    base_x = y_rule["x"] + y_rule["w"] / 2

    y_ticks = calibration_config["y_axis"]["major_ticks"]["bboxes"]
    y_points = [(base_x, t["y"] + t["h"] / 2) for t in y_ticks]

    x_rule = calibration_config["x_axis"]["rule"]["bbox"]
    base_y = x_rule["y"] + x_rule["h"] / 2

    x_ticks = calibration_config["x_axis"]["major_ticks"]["bboxes"]
    x_points = [(t["x"] + t["w"] / 2, base_y) for t in x_ticks]

    return x_points, y_points


def map_pixels_to_actual_values(pixels_list, calibration_config):
    if not pixels_list:
        return []

    y_ticks = calibration_config["y_axis"]["major_ticks"]
    y_bboxes = y_ticks["bboxes"]
    y_vals = [float(v) for v in y_ticks["values"]]
    y_pixs = [b["y"] + b["h"] / 2 for b in y_bboxes]

    poly_y = np.polyfit(y_pixs, y_vals, 1)

    x_ticks_info = calibration_config["x_axis"]["major_ticks"]
    x_ticks_bboxes = x_ticks_info["bboxes"]
    x_labels_text = calibration_config["x_axis"]["major_labels"]["values"]

    unique_x = {}
    for val_text, bbox in zip(x_labels_text, x_ticks_bboxes):
        cx_tick = bbox["x"] + bbox["w"] / 2
        unique_x[round(cx_tick, 1)] = val_text

    x_pix_unique = list(unique_x.keys())

    mapped_points = []
    for pt in pixels_list:
        cx, cy = pt["x"], pt["y"]

        y_val = float(np.polyval(poly_y, cy))

        closest_px = min(x_pix_unique, key=lambda px: abs(cx - px))
        x_val = unique_x[closest_px]

        if isinstance(x_val, str) and x_val.isdigit():
            x_val = int(x_val)

        mapped_points.append({"x": x_val, "y": round(y_val, 3)})

    mapped_points.sort(key=lambda p: float(p["x"]) if str(p["x"]).replace('.', '', 1).isdigit() else str(p["x"]))

    return mapped_points


def export_to_json(mapped_points, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(mapped_points, f, indent=4, ensure_ascii=False)


def export_to_csv(mapped_points, filepath):
    if not mapped_points:
        return
        
    keys = ["x", "y"]
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(mapped_points)