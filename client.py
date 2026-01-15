"""
YOLO-World Client Script

Send video frames to YOLO-World server for inference.
Similar to inference.py but uses client-server architecture.

Usage:
    python client.py --config configs/inference/video_inference.yaml --server-url http://localhost:12182
"""

import argparse
import os
import sys
import time
import requests
from pathlib import Path
from typing import List
from tqdm import tqdm

from inference.config import InferenceConfig
from inference.detection_result import FrameDetections
from inference.utils import (
    get_unique_output_directory,
    save_inference_results,
    load_video_frames,
    send_frame_to_server,
    dict_to_frame_detections
)

def run_client_inference(config: InferenceConfig, server_url: str, text_prompts: List[str] = None):
    """
    Run YOLO-World inference using client-server architecture.

    Args:
        config: InferenceConfig object with all parameters
        server_url: URL of YOLO-World server endpoint
        text_prompts: Optional list of custom text prompts (overrides config)
    """
    input_path = Path(config.input_path)
    input_type = config.input_type
    is_video = input_type == 'video'

    if not is_video:
        raise ValueError("Client currently only supports video input")

    # Use custom prompts if provided, otherwise use config prompts
    prompts_to_use = text_prompts if text_prompts is not None else config.text_prompts

    print(f"Input: {config.input_path}")
    print(f"Type: {input_type}")
    print(f"Server: {server_url}")
    print(f"Text prompts ({len(prompts_to_use)}): {prompts_to_use}")

    # Create subdirectory named after the input file (with timestamp if exists)
    input_stem = input_path.stem
    output_directory = get_unique_output_directory(config.output_directory, input_stem)
    os.makedirs(output_directory, exist_ok=True)
    print(f"Output directory: {output_directory}")

    # Load video
    start_load = time.time()
    frames, fps, total_frames = load_video_frames(config.input_path)
    load_time = time.time() - start_load
    print(f"Input loading time: {load_time:.2f} seconds\n")

    # Check server health
    print("Checking server connection...")
    try:
        requests.get(server_url.replace('/yolo_world', '/'), timeout=5)
        print("✓ Server is reachable")
    except requests.exceptions.RequestException:
        print("✗ Server is not reachable - is it running?")
        print(f"  Expected at: {server_url}")
        sys.exit(1)

    # Process frames
    print("\nSending frames to server for inference...")
    start_inference = time.time()

    all_detections = []

    for frame_id, frame in enumerate(tqdm(frames, desc="Processing frames")):
        try:
            result_dict = send_frame_to_server(
                frame,
                server_url,
                text_prompts=prompts_to_use
            )
            frame_detections = dict_to_frame_detections(result_dict, frame_id)
            all_detections.append(frame_detections)
        except Exception as e:
            print(f"\nError processing frame {frame_id}: {e}")
            # Create empty detection for this frame
            frame_detections = FrameDetections(
                frame_id=frame_id,
                detections=[],
                frame_width=frame.shape[1],
                frame_height=frame.shape[0]
            )
            all_detections.append(frame_detections)

    inference_time = time.time() - start_inference

    print(f"\nInference completed in {inference_time:.2f} seconds")
    print(f"Average time per frame: {inference_time / len(frames):.4f} seconds")
    print(f"Inference FPS: {len(frames) / inference_time:.2f}")

    total_detections = sum(len(fd.detections) for fd in all_detections)
    frames_with_detections = sum(1 for fd in all_detections if len(fd.detections) > 0)
    print(f"\nTotal detections: {total_detections}")
    print(f"Frames with detections: {frames_with_detections}/{len(frames)}")
    print(f"Average detections per frame: {total_detections / len(frames):.2f}")

    # Save all results using shared function
    save_inference_results(
        frames=frames,
        all_detections=all_detections,
        output_directory=output_directory,
        config=config,
        input_path=config.input_path,
        input_type=input_type,
        fps=fps,
        total_frames=total_frames,
        inference_time=inference_time,
        text_prompts=prompts_to_use,
        gpu_memory_summary={"note": "GPU memory tracked on server side"},
        model_size_mb=0.0,  # Not available on client side
        image_filenames=None
    )

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="YOLO-World Client - Send video to server for inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run client with video (uses prompts from config file)
  python client.py --config configs/inference/video_inference.yaml --server-url http://localhost:12182/yolo_world

  # Override prompts from command line
  python client.py --config configs/inference/video_inference.yaml --prompts "person,car,dog,cat"

  # Use custom server port
  python client.py --config configs/inference/video_inference.yaml --server-url http://localhost:5000/yolo_world
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--server-url',
        type=str,
        default='http://localhost:12182/yolo_world',
        help='YOLO-World server endpoint URL (default: http://localhost:12182/yolo_world)'
    )
    parser.add_argument(
        '--prompts',
        type=str,
        help='Comma-separated list of class names to detect (overrides config file prompts)'
    )

    args = parser.parse_args()

    # Parse prompts if provided
    custom_prompts = None
    if args.prompts:
        custom_prompts = [p.strip() for p in args.prompts.split(',')]

    try:
        if not os.path.exists(args.config):
            print(f"Error: Configuration file not found: {args.config}", file=sys.stderr)
            sys.exit(1)

        print(f"Loading configuration from: {args.config}")
        config = InferenceConfig(args.config)
        print(config)

        run_client_inference(config, args.server_url, text_prompts=custom_prompts)

    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
