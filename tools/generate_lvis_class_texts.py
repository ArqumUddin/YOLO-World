import json
import argparse
from mmdet.datasets import LVISV1Dataset

def generate_lvis_class_texts(ann_file, out_path):
    print(f"Loading annotations from {ann_file}...")
    # Dictionary to map id to name
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    categories = data.get('categories', [])
    sorted_cats = sorted(categories, key=lambda x: x['id'])
    class_texts = []
    for cat in sorted_cats:
        name = cat['name']
        name = name.replace('_', ' ')
        class_texts.append([name])
    print(f"Generated {len(class_texts)} class texts.")
    with open(out_path, 'w') as f:
        json.dump(class_texts, f, indent=4)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate LVIS class texts json')
    parser.add_argument('--ann_file', type=str, required=True, help='Path to LVIS annotation json (e.g. lvis_v1_minival.json)')
    parser.add_argument('--out', type=str, required=True, help='Output path (e.g. data/texts/lvis_v1_class_texts.json)')
    args = parser.parse_args()
    
    generate_lvis_class_texts(args.ann_file, args.out)
