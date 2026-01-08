import json
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def main():
    # Robust path detection ensuring we find sdg_dataset relative to this script
    script_dir = Path(__file__).parent.resolve()
    yolo_world_root = script_dir.parent
    project_root = yolo_world_root.parent

    # Setup data directory inside YOLO-World
    data_dir = yolo_world_root / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Link SDG dataset (Tiamat/sdg_dataset)
    sdg_source = project_root / 'sdg_dataset'
    sdg_target = data_dir / 'sdg'
    
    if not sdg_source.exists():
        print(f"Error: SDG dataset not found at {sdg_source}")
        return
        
    if not sdg_target.exists():
        print(f"Linking {sdg_source} to {sdg_target}")
        os.symlink(sdg_source, sdg_target)
    else:
        print(f"Found existing link/dir at {sdg_target}")
        
    # Generate texts
    texts_dir = data_dir / 'texts'
    texts_dir.mkdir(exist_ok=True)
    
    dataset_dir = sdg_target / 'dataset'
    ann_file = dataset_dir / '_annotations.coco.json'
    out_path = texts_dir / 'sdg_texts.json'
    
    if not ann_file.exists():
        print(f"Annotation file not found yet at {ann_file}. Training configuration will fail until generation is complete.")
        return

    print(f"Loading annotations from {ann_file}...")
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # --- SPLIT BY SCENE ---
    print(" performing SCENE-BASED Split (80% Train, 10% Val, 10% Test)...")
    images = data['images']
    annotations = data['annotations']
    categories = data.get('categories', [])
    
    # Group images by scene
    # Filename format: {scene_id}_surf{surf_idx}_view_{view_idx}.png
    scene_to_images = defaultdict(list)
    for img in images:
        filename = img['file_name']
        if '_surf' in filename:
            scene_id = filename.split('_surf')[0]
        else:
            # Fallback if naming convention changes, assume single scene or hash
            scene_id = 'unknown'
        scene_to_images[scene_id].append(img)
        
    scene_ids = sorted(list(scene_to_images.keys()))
    random.seed(42) # Ensure deterministic split
    random.shuffle(scene_ids)
    
    num_scenes = len(scene_ids)
    num_train = int(num_scenes * 0.80)
    num_val = int(num_scenes * 0.10)
    
    train_scenes = set(scene_ids[:num_train])
    val_scenes = set(scene_ids[num_train:num_train + num_val])
    test_scenes = set(scene_ids[num_train + num_val:])
    
    print(f"  Total Scenes: {num_scenes}")
    print(f"  Train Scenes: {len(train_scenes)}")
    print(f"  Val Scenes:   {len(val_scenes)}")
    print(f"  Test Scenes:  {len(test_scenes)}")
    
    # Helper to build split
    def build_split(target_scenes, split_name):
        split_images = []
        split_img_ids = set()
        
        for scene in target_scenes:
            imgs = scene_to_images[scene]
            split_images.extend(imgs)
            for img in imgs:
                split_img_ids.add(img['id'])
                
        split_anns = [ann for ann in annotations if ann['image_id'] in split_img_ids]
        
        split_data = {
            'images': split_images,
            'annotations': split_anns,
            'categories': categories
        }
        
        out_file = dataset_dir / f"{split_name}.json"
        with open(out_file, 'w') as f:
            json.dump(split_data, f)
        print(f"  Saved {split_name} split to {out_file} ({len(split_images)} images, {len(split_anns)} annotations)")
        
    build_split(train_scenes, 'train')
    build_split(val_scenes, 'val')
    build_split(test_scenes, 'test')
    
    # --- END SPLIT ---

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
    main()
