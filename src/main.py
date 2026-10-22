import argparse
import json
import os
import cv2
import matplotlib.pyplot as plt

from converter import export_to_csv, export_to_json, map_pixels_to_actual_values
from detector import detect_dots


def _sort_key(p):
    val = p["x"]
    try:
        return (0, float(val))
    except (ValueError, TypeError):
        return (1, str(val))


def visualize_results(
    image_bgr, mapped_points, gt_points=None, output_path="comparison_result.png"
):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Oryginalny obraz wejściowy", fontsize=14)
    axes[0].axis("off")

    axes[1].set_title("Odtworzony wykres z danych (GT vs Pred)", fontsize=14)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    if gt_points:
        gts_sorted = sorted(gt_points, key=_sort_key)
        gt_x = [str(pt["x"]) for pt in gts_sorted]
        gt_y = [pt["y"] for pt in gts_sorted]

        axes[1].plot(
            gt_x,
            gt_y,
            "o--",
            color="red",
            label="GT (JSON)",
            markersize=6,
            linewidth=1.5,
            alpha=0.7,
        )

    if mapped_points:
        preds_sorted = sorted(mapped_points, key=_sort_key)
        pred_x = [str(pt["x"]) for pt in preds_sorted]
        pred_y = [pt["y"] for pt in preds_sorted]

        axes[1].plot(
            pred_x,
            pred_y,
            "o-",
            color="limegreen",
            label="Pred",
            markersize=5,
            linewidth=2,
        )
    else:
        if not gt_points:
            axes[1].text(
                0.5,
                0.5,
                "Nie wykryto żadnych punktów",
                ha="center",
                va="center",
                fontsize=12,
            )

    axes[1].legend(fontsize=10, loc="upper right")
    axes[1].tick_params(axis="x", rotation=45, labelsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Ekstrakcja danych z wykresów liniowych (dot-line)."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Ścieżka do obrazu wejściowego (.jpg/.png)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Ścieżka do pliku JSON z kalibracją osi",
    )
    parser.add_argument(
        "--out_csv", type=str, default="output.csv", help="Ścieżka zapisu CSV"
    )
    parser.add_argument(
        "--out_json", type=str, default="output.json", help="Ścieżka zapisu JSON"
    )
    args = parser.parse_args()

    print(f"--- Uruchamianie ekstrakcji dla: {args.image} ---")

    image = cv2.imread(args.image)
    if image is None:
        print(f"BŁĄD: Nie można wczytać obrazu: {args.image}")
        return

    matched_sample = None
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        if isinstance(config_data, dict) and "samples" in config_data:
            image_filename = os.path.basename(args.image)
            image_id_str = os.path.splitext(image_filename)[0]

            matched_sample = next(
                (
                    s
                    for s in config_data["samples"]
                    if str(s.get("image_index")) == image_id_str
                ),
                None,
            )

            if matched_sample and "general_figure_info" in matched_sample:
                calibration_config = matched_sample["general_figure_info"]
            else:
                print(
                    f"BŁĄD: Nie znaleziono adnotacji dla obrazu ID {image_id_str} w pliku {args.config}"
                )
                return

        elif isinstance(config_data, dict) and "general_figure_info" in config_data:
            calibration_config = config_data["general_figure_info"]
            matched_sample = config_data
        elif isinstance(config_data, dict) and "y_axis" in config_data:
            calibration_config = config_data
        else:
            print("BŁĄD: Niepoprawna struktura pliku konfiguracyjnego.")
            return

    except Exception as e:
        print(f"BŁĄD: Nie można odczytać pliku konfiguracyjnego {args.config}: {e}")
        return

    x_labels = []
    if calibration_config and "x_axis" in calibration_config:
        x_labels = (
            calibration_config["x_axis"]
            .get("major_labels", {})
            .get("values", [])
        )
        if not x_labels:
            x_labels = (
                calibration_config["x_axis"]
                .get("major_ticks", {})
                .get("values", [])
            )

    gt_points = []
    models_data = None
    if matched_sample and "models" in matched_sample:
        models_data = matched_sample["models"]
    elif isinstance(config_data, dict) and "models" in config_data:
        models_data = config_data["models"]

    if models_data:
        for model in models_data:
            x_vals = model.get("x", [])
            y_vals = model.get("y", [])
            for gx, gy in zip(x_vals, y_vals):
                if isinstance(gx, int) or (isinstance(gx, str) and gx.isdigit()):
                    idx = int(gx)
                    if x_labels and 0 <= idx < len(x_labels):
                        gx = x_labels[idx]

                try:
                    gt_points.append({"x": gx, "y": float(gy)})
                except (ValueError, TypeError):
                    continue

    print("1. Trwa detekcja markerów na obrazie...")
    pixels_list = detect_dots(image)
    print(f"   -> Znaleziono {len(pixels_list)} punktów.")

    print("2. Trwa przeliczanie pikseli na wartości osi...")
    mapped_points = map_pixels_to_actual_values(
        pixels_list, calibration_config
    )

    print(f"3. Zapisywanie wyników do {args.out_csv} i {args.out_json}...")
    export_to_csv(mapped_points, args.out_csv)
    export_to_json(mapped_points, args.out_json)

    print("4. Generowanie wizualizacji porównawczej...")
    visualize_results(
        image, mapped_points, gt_points=gt_points, output_path="comparison_result.png"
    )

    print("--- Sukces, zapisano wyniki do pliku ---")


if __name__ == "__main__":
    main()