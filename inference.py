"""
YOLO-World Inference Script

Run object detection on images, videos, or directories of images using YOLO-World model.
Outputs annotated media and results.json with detection data.

Usage:
    # Run inference on video, single image, or directory of images
    python inference.py --config inference/configs/yolo_world_v2_x_inference.yaml

    # Print summary of existing results
    python inference.py --print-summary path/to/results.json
"""

import argparse
import os
import sys
import time
import torch
from pathlib import Path
import traceback

from inference.config import InferenceConfig
from inference.gpu_memory_tracker import GPUMemoryTracker
from inference.yolo_world import YOLOWorld
from inference.results_writer import ResultsWriter
from inference.utils import (
    prepare_inference_input,
    save_inference_results
)

def run_inference(config: InferenceConfig):
    """
    Run YOLO-World inference using configuration.

    Args:
        config: InferenceConfig object with all parameters
    """
    # Load and prepare input (shared preprocessing logic)
    input_data = prepare_inference_input(config)

    gpu_tracker = GPUMemoryTracker(device=config.device)
    if gpu_tracker.cuda_available:
        torch.cuda.reset_peak_memory_stats()

    print("Initializing YOLO-World model...")
    model = YOLOWorld(
        config_path=config.config_file,
        checkpoint_path=config.checkpoint,
        text_prompts=config.text_prompts,
        device=config.device,
        confidence_threshold=config.confidence_threshold,
        nms_threshold=config.nms_threshold,
        max_detections=config.max_detections
    )

    model_size_mb = model.get_model_size_mb()
    print(f"Model size: {model_size_mb:.2f} MB")

    print("Running inference...")
    start_inference = time.time()

    all_detections = model.predict_frames(input_data.frames, start_frame_id=0)
    gpu_tracker.record_snapshot()

    inference_time = time.time() - start_inference

    print(f"\nInference completed in {inference_time:.2f} seconds")
    print(f"Average time per frame: {inference_time / len(input_data.frames):.4f} seconds")
    print(f"Inference FPS: {len(input_data.frames) / inference_time:.2f}")

    total_detections = sum(len(fd.detections) for fd in all_detections)
    frames_with_detections = sum(1 for fd in all_detections if len(fd.detections) > 0)
    print(f"\nTotal detections: {total_detections}")
    print(f"Frames with detections: {frames_with_detections}/{len(input_data.frames)}")
    print(f"Average detections per frame: {total_detections / len(input_data.frames):.2f}")

    # Save all results using shared function
    gpu_memory_summary = gpu_tracker.get_summary()

    save_inference_results(
        frames=input_data.frames,
        all_detections=all_detections,
        output_directory=input_data.output_directory,
        config=config,
        input_path=config.input_path,
        input_type=input_data.input_type,
        fps=input_data.fps,
        total_frames=input_data.total_frames,
        inference_time=inference_time,
        text_prompts=config.text_prompts,
        gpu_memory_summary=gpu_memory_summary,
        model_size_mb=model_size_mb,
        image_filenames=input_data.image_filenames
    )

def run_from_config(config_path: str):
    """
    Run inference from YAML configuration file.

    Args:
        config_path: Path to YAML config file
    """
    print(f"Loading configuration from: {config_path}")
    config = InferenceConfig(config_path)
    print(config)
    run_inference(config)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="YOLO-World Inference Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Run inference
            python inference.py --config inference/configs/video_inference.yaml

            # Print summary of existing results
            python inference.py --print-summary results/results.json
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file for inference'
    )
    group.add_argument(
        '--print-summary',
        type=str,
        metavar='RESULTS_FILE',
        help='Print summary of existing results JSON file'
    )

    args = parser.parse_args()

    try:
        if args.config:
            if not os.path.exists(args.config):
                print(f"Error: Configuration file not found: {args.config}", file=sys.stderr)
                sys.exit(1)
            run_from_config(args.config)

        elif args.print_summary:
            if not os.path.exists(args.print_summary):
                print(f"Error: Results file not found: {args.print_summary}", file=sys.stderr)
                sys.exit(1)
            ResultsWriter.print_summary(args.print_summary)

    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
