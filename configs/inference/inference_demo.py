import argparse
import sys
import os
import glob
from pathlib import Path
import cv2
import torch
import numpy as np

# Add YOLO-World root to python path to allow imports of 'inference' package
current_file = Path(__file__).resolve()
repo_root = current_file.parents[2]
sys.path.append(str(repo_root))

from inference.yolo_world import YOLOWorld
from inference.visualization.annotator import FrameAnnotator
from inference.detection_result import Detection

def get_ycb_classes(ycb_dir):
    """
    Scan the YCB config directory to get the list of class names.
    """
    ycb_configs_dir = Path(ycb_dir) / "configs"
    if not ycb_configs_dir.exists():
        print(f"Warning: YCB configs not found at {ycb_configs_dir}")
        return []

    classes = []
    for config_file in ycb_configs_dir.glob("*.object_config.json"):
        # filename format: 002_master_chef_can.object_config.json
        # we want: master_chef_can
        stem = config_file.stem.replace('.object_config', '')
        # Remove the leading number prefix (e.g., '002_')
        parts = stem.split('_')
        if len(parts) > 1 and parts[0].isdigit():
            class_name = '_'.join(parts[1:])
        else:
            class_name = stem
        
        # Replace underscores with spaces for better text embedding matching? 
        # Usually exact match with spaces is better for CLIP text encoder
        class_name = class_name.replace('_', ' ')
        classes.append(class_name)
    
    return sorted(list(set(classes)))

def main():
    parser = argparse.ArgumentParser(description="YOLO-World Inference Demo")
    
    # Defaults based on workspace analysis
    default_config = repo_root / "configs/inference/yolo_world_v2_l.py"
    # Looking for weights - assuming they are in weights/ directory
    default_checkpoint = repo_root / "weights/yolo_world_v2_l_stage2.pth" 
    # Fallback to stage1 if stage2 not present? Let's point to user to verify
    
    # Sample image
    default_image = "/home/jz4019/Tiamat/sdg_output/images/102344250_surf0_view_0001.png"
    
    parser.add_argument("--config", default=str(default_config), help="Path to model config")
    parser.add_argument("--checkpoint", default=str(default_checkpoint), help="Path to model checkpoint")
    parser.add_argument("--image", default=default_image, help="Path to input image")
    parser.add_argument("--mode", choices=["empty", "ycb"], default="ycb", help="Text prompt mode: 'empty' or 'ycb'")
    parser.add_argument("--output-dir", default="demo_outputs", help="Directory to save results")
    parser.add_argument("--score-thr", type=float, default=0.1, help="Score threshold")
    parser.add_argument("--ycb-dir", default="/home/jz4019/Tiamat/data/ycb", help="Path to YCB dataset root")
    
    args = parser.parse_args()
    
    # Calculate text prompts
    if args.mode == "empty":
        print("Mode: Empty text input")
        text_prompts = [] 
    else:
        print("Mode: YCB Label Set")
        text_prompts = get_ycb_classes(args.ycb_dir)
        print(f"Loaded {len(text_prompts)} YCB classes: {text_prompts[:5]} ...")

    # Initialize model
    try:
        model = YOLOWorld(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            text_prompts=text_prompts,
            confidence_threshold=args.score_thr,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("Please ensure config and checkpoint paths are correct.")
        return

    # Load image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return
        
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Failed to load image {image_path}")
        return
        
    print(f"Running inference on {image_path.name}...")
    
    # Run prediction
    try:
        # Note: 'prompts' argument in predict is optional and overrides init prompts if provided.
        # We rely on init prompts here.
        detections = model.predict(image)
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Visualize
    annotator = FrameAnnotator()
    annotated_frame = annotator.annotate_frame(image, detections)
    
    # Save output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / f"result_{args.mode}_{image_path.name}"
    cv2.imwrite(str(output_path), annotated_frame)
    
    print(f"Saved visualization to {output_path}")
    print(f"Found {len(detections.detections)} objects.")
    
    # Print top detections
    if detections.detections:
        print("\nTop detections:")
        sorted_dets = sorted(detections.detections, key=lambda x: x.confidence, reverse=True)
        for i, det in enumerate(sorted_dets[:10]):
            print(f"  {i+1}. {det.class_name} ({det.confidence:.2f})")

if __name__ == "__main__":
    main()
