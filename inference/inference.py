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
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple
import traceback
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import InferenceConfig
from inference.yolo_world import YOLOWorld
from visualization.annotator import FrameAnnotator
from visualization.video_writer import AnnotatedVideoWriter
from results_writer import ResultsWriter
from gpu_memory_tracker import GPUMemoryTracker


def load_video_frames(video_path: str) -> Tuple[List[np.ndarray], float, int]:
    """
    Load all frames from a video file.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (frames, fps, total_frames)
    """
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps:.2f}, Total frames: {total_frames}")

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_count += 1

    cap.release()
    print(f"Loaded {len(frames)} frames from video")

    return frames, fps, total_frames


def load_image(image_path: str) -> np.ndarray:
    """
    Load a single image.

    Args:
        image_path: Path to image file

    Returns:
        RGB image as numpy array
    """
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def load_images_from_directory(directory_path: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load all images from a directory.

    Args:
        directory_path: Path to directory containing images

    Returns:
        Tuple of (images list, image filenames list)
    """
    print(f"Loading images from directory: {directory_path}")

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []

    # Get all image files from directory
    for filename in sorted(os.listdir(directory_path)):
        if os.path.splitext(filename)[1].lower() in image_extensions:
            image_files.append(filename)

    if not image_files:
        raise ValueError(f"No image files found in directory: {directory_path}")

    print(f"Found {len(image_files)} images")

    images = []
    loaded_filenames = []

    for filename in image_files:
        image_path = os.path.join(directory_path, filename)
        try:
            image = cv2.imread(image_path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image_rgb)
                loaded_filenames.append(filename)
            else:
                print(f"Warning: Could not load {filename}, skipping...")
        except Exception as e:
            print(f"Warning: Error loading {filename}: {e}, skipping...")

    print(f"Successfully loaded {len(images)} images")

    return images, loaded_filenames


def run_inference(config: InferenceConfig):
    """
    Run YOLO-World inference using configuration.

    Args:
        config: InferenceConfig object with all parameters
    """
    os.makedirs(config.output_directory, exist_ok=True)

    input_path = Path(config.input_path)
    input_type = config.input_type
    is_video = input_type == 'video'
    is_directory = input_type == 'directory'

    print(f"Input: {config.input_path}")
    print(f"Type: {input_type}")
    print(f"Output directory: {config.output_directory}")

    start_load = time.time()
    image_filenames = None  # Track filenames for directory mode

    if is_video:
        frames, fps, total_frames = load_video_frames(config.input_path)
    elif is_directory:
        frames, image_filenames = load_images_from_directory(config.input_path)
        fps = None
        total_frames = len(frames)
    else:
        frames = [load_image(config.input_path)]
        fps = None
        total_frames = 1
    load_time = time.time() - start_load
    print(f"Input loading time: {load_time:.2f} seconds\n")

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

    from tqdm import tqdm
    all_detections = []
    for i, frame in enumerate(tqdm(frames, desc="Processing frames")):
        frame_det = model.predict_frame(frame, frame_id=i)
        all_detections.append(frame_det)
        gpu_tracker.record_snapshot()

    inference_time = time.time() - start_inference

    print(f"\nInference completed in {inference_time:.2f} seconds")
    print(f"Average time per frame: {inference_time / len(frames):.4f} seconds")
    print(f"Inference FPS: {len(frames) / inference_time:.2f}")

    total_detections = sum(len(fd.detections) for fd in all_detections)
    frames_with_detections = sum(1 for fd in all_detections if len(fd.detections) > 0)
    print(f"\nTotal detections: {total_detections}")
    print(f"Frames with detections: {frames_with_detections}/{len(frames)}")
    print(f"Average detections per frame: {total_detections / len(frames):.2f}")

    annotator = FrameAnnotator(
        show_confidence=config.show_confidence,
        show_class_name=config.show_class_name,
        bbox_thickness=config.bbox_thickness,
        font_scale=config.font_scale
    )

    if config.save_annotated_frames:
        print("\nSaving annotated frames...")
        os.makedirs(config.annotated_frames_dir, exist_ok=True)

        from tqdm import tqdm
        for idx, (frame, frame_det) in enumerate(tqdm(
            zip(frames, all_detections),
            desc="Saving frames",
            total=len(frames)
        )):
            annotated_frame = annotator.annotate_frame(frame, frame_det)

            # Use original filename for directory mode, otherwise use frame numbering
            if is_directory and image_filenames:
                filename = f"{Path(image_filenames[idx]).stem}_annotated{Path(image_filenames[idx]).suffix}"
            else:
                filename = f"frame_{frame_det.frame_id:06d}.jpg"
            output_path = os.path.join(config.annotated_frames_dir, filename)

            annotated_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, annotated_bgr)

        print(f"Annotated frames saved to: {config.annotated_frames_dir}")

    if config.save_annotated_video and is_video:
        print("\nGenerating annotated video...")
        output_video_path = os.path.join(
            config.output_directory,
            f"{Path(config.input_path).stem}_annotated.mp4"
        )

        with AnnotatedVideoWriter(output_video_path, fps=fps) as writer:
            for frame, frame_det in zip(frames, all_detections):
                annotated_frame = annotator.annotate_frame(frame, frame_det)
                writer.write_frame(annotated_frame)

        print(f"Annotated video saved to: {output_video_path}")

    if config.save_annotated_video and not is_video and not is_directory:
        print("\nGenerating annotated image...")
        output_image_path = os.path.join(
            config.output_directory,
            f"{Path(config.input_path).stem}_annotated.jpg"
        )

        annotated_frame = annotator.annotate_frame(frames[0], all_detections[0])

        annotated_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image_path, annotated_bgr)
        print(f"Annotated image saved to: {output_image_path}")

    if config.save_annotated_video and is_directory:
        print("\nGenerating annotated images for directory...")
        annotated_images_dir = os.path.join(config.output_directory, "annotated_images")
        os.makedirs(annotated_images_dir, exist_ok=True)

        for idx, (frame, frame_det) in enumerate(tqdm(
            zip(frames, all_detections),
            desc="Saving annotated images",
            total=len(frames)
        )):
            annotated_frame = annotator.annotate_frame(frame, frame_det)

            # Use original filename
            if image_filenames:
                filename = f"{Path(image_filenames[idx]).stem}_annotated{Path(image_filenames[idx]).suffix}"
            else:
                filename = f"image_{idx:06d}_annotated.jpg"
            output_path = os.path.join(annotated_images_dir, filename)

            annotated_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, annotated_bgr)

        print(f"Annotated images saved to: {annotated_images_dir}")

    if config.save_results_json:
        print("\nSaving results to JSON...")
        output_json_path = os.path.join(config.output_directory, config.results_filename)

        gpu_memory_summary = gpu_tracker.get_summary()

        ResultsWriter.write_results(
            output_path=output_json_path,
            model_config=config.config_file,
            model_checkpoint=config.checkpoint,
            input_path=config.input_path,
            all_frame_detections=all_detections,
            text_prompts=config.text_prompts,
            execution_time=inference_time,
            fps=fps,
            total_frames=total_frames,
            gpu_memory=gpu_memory_summary,
            model_name=config.model_name,
            display_name=config.display_name,
            model_size_mb=model_size_mb
        )

        ResultsWriter.print_summary(output_json_path)


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
