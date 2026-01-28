"""
YOLO-World Client Script

Send video frames to YOLO-World server for inference.
Similar to inference.py but uses client-server architecture.

Usage:
    python client.py --config configs/inference/client_server/client.yaml --server-url http://localhost:12182
"""

import argparse
import os
import sys
import time
import requests
from pathlib import Path
from typing import List
from tqdm import tqdm

from inference.config import ClientConfig
from inference.detection_result import FrameDetections
from inference.utils import (
    prepare_inference_input,
    save_inference_results,
    send_frame_to_server,
    dict_to_frame_detections
)

def run_client_inference(config: ClientConfig, server_url: str, text_prompts: List[str] = None):
    """
    Run YOLO-World inference using client-server architecture.

    Args:
        config: ClientConfig object with input/output parameters
        server_url: URL of YOLO-World server endpoint
        text_prompts: Optional list of custom text prompts (CLI only)
    """
    # Only send prompts when explicitly provided via CLI
    prompts_to_use = text_prompts if text_prompts else None

    # Load and prepare input (shared preprocessing logic)
    input_data = prepare_inference_input(config)

    print(f"Server: {server_url}")
    if prompts_to_use:
        print(f"Text prompts (CLI) ({len(prompts_to_use)}): {prompts_to_use}")
    else:
        print("Text prompts: <none> (server defaults)")

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

    for frame_id, frame in enumerate(tqdm(input_data.frames, desc="Processing frames")):
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
    print(f"Average time per frame: {inference_time / len(input_data.frames):.4f} seconds")
    print(f"Inference FPS: {len(input_data.frames) / inference_time:.2f}")

    total_detections = sum(len(fd.detections) for fd in all_detections)
    frames_with_detections = sum(1 for fd in all_detections if len(fd.detections) > 0)
    print(f"\nTotal detections: {total_detections}")
    print(f"Frames with detections: {frames_with_detections}/{len(input_data.frames)}")
    print(f"Average detections per frame: {total_detections / len(input_data.frames):.2f}")

    # Save all results using shared function
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
        text_prompts=prompts_to_use,
        gpu_memory_summary={"note": "GPU memory tracked on server side"},
        model_size_mb=0.0,  # Not available on client side
        image_filenames=input_data.image_filenames
    )

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="YOLO-World Client - Send video to server for inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run client with video (server uses its default prompts)
  python client.py --config configs/inference/client_server/client.yaml --server-url http://localhost:12182/yolo_world

  # Override prompts from command line
  python client.py --config configs/inference/client_server/client.yaml --prompts "person,car,dog,cat"

  # Use custom server port
  python client.py --config configs/inference/client_server/client.yaml --server-url http://localhost:5000/yolo_world
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
        default=None,
        help='YOLO-World server endpoint URL (overrides config)'
    )
    parser.add_argument(
        '--prompts',
        type=str,
        help='Comma-separated list of class names to detect (CLI only)'
    )

    args = parser.parse_args()

    # Parse prompts if provided
    custom_prompts = None
    if args.prompts:
        custom_prompts = [p.strip() for p in args.prompts.split(',') if p.strip()]
        if not custom_prompts:
            custom_prompts = None

    try:
        if not os.path.exists(args.config):
            print(f"Error: Configuration file not found: {args.config}", file=sys.stderr)
            sys.exit(1)

        print(f"Loading configuration from: {args.config}")
        config = ClientConfig(args.config)
        print(config)

        server_url = args.server_url or config.server_url
        run_client_inference(config, server_url, text_prompts=custom_prompts)

    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
