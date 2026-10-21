import argparse
import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def is_monochrome(img, sat_max_threshold=40):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 1].max() < sat_max_threshold


def detect_dots(img):
    h, w, _ = img.shape
    detected_centers = []

    if not is_monochrome(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]

        _, thresh_sat = cv2.threshold(sat, 40, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh_clean = cv2.morphologyEx(
            thresh_sat, cv2.MORPH_OPEN, kernel, iterations=1
        )

        contours, _ = cv2.findContours(
            thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue

            circularity = (4 * np.pi * area) / (perimeter**2)

            if 15 <= area <= 400 and circularity >= 0.40:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    detected_centers.append({"x": cx, "y": cy})
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=3,
        )

        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        horizontal_lines = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, hor_kernel, iterations=1
        )
        y_indices, _ = np.where(horizontal_lines > 0)

        if len(y_indices) > 0:
            top_grid_y = np.min(y_indices)
            thresh[0 : max(0, top_grid_y - 2), :] = 0

            bottom_grid_y = np.max(y_indices)
            thresh[min(h, bottom_grid_y + 4) :, :] = 0

        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        vertical_lines = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, ver_kernel, iterations=1
        )
        _, x_indices = np.where(vertical_lines > 0)

        if len(x_indices) > 0:
            left_grid_x = np.min(x_indices)
            thresh[:, 0 : max(0, left_grid_x - 4)] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh_clean = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, kernel, iterations=1
        )

        contours, _ = cv2.findContours(
            thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue

            x, y, bw, bh = cv2.boundingRect(c)
            aspect_ratio = float(bw) / bh if bh > 0 else 0
            circularity = (4 * np.pi * area) / (perimeter**2)

            if (
                15 <= area <= 350
                and circularity >= 0.40
                and 0.65 <= aspect_ratio <= 1.45
            ):
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    detected_centers.append({"x": cx, "y": cy})

    detected_centers.sort(key=lambda p: p["x"])
    return detected_centers


def save_results(points, output_path, format_type):
    if format_type == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(points, f, indent=4)
    elif format_type == "csv":
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("x,y\n")
            for pt in points:
                f.write(f"{pt['x']},{pt['y']}\n")


def visualize_and_save(img, points, viz_path):
    img_result = img.copy()

    for pt in points:
        cx, cy = pt["x"], pt["y"]
        cv2.circle(img_result, (cx, cy), 6, (0, 0, 255), 2)
        cv2.circle(img_result, (cx, cy), 2, (0, 255, 0), -1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Obraz Wejściowy", fontsize=12)
    ax1.axis("off")

    ax2.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"Detekcja Kropek (Wykryto: {len(points)})", fontsize=12)
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(viz_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="detektor kropek"
    )
    parser.add_argument(
        "--image", required=True, help="Ścieżka do obrazu wykresu"
    )
    parser.add_argument(
        "--output", default="output", help="Ścieżka/Nazwa pliku wyjściowego"
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Format zapisu wyników (json/csv)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Błąd: Plik {args.image} nie istnieje!")
        return

    img = cv2.imread(args.image)

    points = detect_dots(img)

    out_file = f"{args.output}.{args.format}"
    save_results(points, out_file, args.format)

    viz_file = f"{args.output}_viz.png"
    visualize_and_save(img, points, viz_file)

    print(f"wykryto {len(points)} kropek.")
    print(f"wyniki zapisano do: {out_file}")
    print(f"wizualizacja zapisano do: {viz_file}")


if __name__ == "__main__":
    main()