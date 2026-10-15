import os
import json
import shutil
import sys


DEFAULT_RAW_DIR = os.path.join('data', 'raw_data')
RAW_DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RAW_DIR

IMAGES_DIR = os.path.join(RAW_DATA_DIR, 'images', 'png')
ANNOTATIONS_JSON_PATH = os.path.join(RAW_DATA_DIR, 'annotations', 'annotations.json')

TARGET_DIR = os.path.join("data", "sample_images")
os.makedirs(TARGET_DIR, exist_ok=True)


with open(ANNOTATIONS_JSON_PATH, "r", encoding='utf-8') as f:
    all_annotations = json.load(f)

selected_samples = []

for item in all_annotations:
    if item.get("type") == 'dot_line':
        selected_samples.append(item)
        if len(selected_samples) == 30:
            break


dev_set = selected_samples[:10]
test_set = selected_samples[10:30]

dataset_metadata = {
    'dataset_name': "PlotQA",
    'original_split': "validation", 
    'selection_criteria': "30 pierwszych chronologicznie wykresów typu 'dot_line' ze zbioru walidacyjnego PlotQA",
    'reproducibility': "Uruchom skrypt download_data.py z podaniem ścieżki do rozpakowanego folderu PlotQA validation jako argumentu",
    'dev_ids': [item['image_index'] for item in dev_set],
    'test_ids': [item['image_index'] for item in test_set], 
    'samples': selected_samples
}

with open(os.path.join('data', 'sample_annotations.json'), 'w', encoding='utf-8') as f:
    json.dump(dataset_metadata, f, indent=4, ensure_ascii=False)

copied_count = 0 
for item in selected_samples:
    img_idx = item['image_index']
    img_name = f"{img_idx}.png"

    src_path = os.path.join(IMAGES_DIR, img_name)
    dst_path = os.path.join(TARGET_DIR, img_name)

    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        copied_count += 1
    else:
        print(f"Nie znaleziono {img_name} w {IMAGES_DIR}")

print(f"{copied_count}/30 plikow skopiowanych")


