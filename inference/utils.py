"""
Utility functions for YOLO-World inference.
"""

import os
import cv2
import base64
import time
import numpy as np
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, NamedTuple
from tqdm import tqdm

from .visualization.annotator import FrameAnnotator
from .visualization.video_writer import AnnotatedVideoWriter
from .results_writer import ResultsWriter
from .detection_result import FrameDetections, Detection, BoundingBox

class InferenceInput(NamedTuple):
    """Container for loaded inference input data."""
    frames: List[np.ndarray]
    fps: Optional[float]
    total_frames: int
    image_filenames: Optional[List[str]]
    output_directory: str
    input_type: str
    input_path: Path
    load_time: float


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


def load_images_from_list(base_dir: str, file_names: List[str]) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load images from a list of file names, resolving relative paths against base_dir.

    Args:
        base_dir: Base directory for relative file paths
        file_names: List of file names or relative paths

    Returns:
        Tuple of (images list, image filenames list)
    """
    print(f"Loading {len(file_names)} images using COCO file list...")

    images = []
    loaded_filenames = []

    for file_name in file_names:
        if not file_name:
            continue
        image_path = file_name if os.path.isabs(file_name) else os.path.join(base_dir, file_name)
        try:
            image = cv2.imread(image_path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image_rgb)
                loaded_filenames.append(file_name)
            else:
                print(f"Warning: Could not load {file_name}, skipping...")
        except Exception as e:
            print(f"Warning: Error loading {file_name}: {e}, skipping...")

    print(f"Successfully loaded {len(images)} images")

    return images, loaded_filenames


def prepare_inference_input(config) -> InferenceInput:
    """
    Prepare input for inference: load frames/images and create output directory.

    This function handles all input preprocessing for both inference.py and client.py,
    eliminating code duplication.

    Args:
        config: InferenceConfig object with input_path, input_type, output_directory

    Returns:
        InferenceInput object containing all loaded data and metadata
    """
    input_path = Path(config.input_path)
    load_path = Path(config.input_path)
    input_type = config.input_type
    is_video = input_type == 'video'
    is_directory = input_type == 'directory'

    # Create subdirectory named after the input file (with timestamp if exists)
    input_stem = input_path.stem
    output_directory = get_unique_output_directory(config.output_directory, input_stem)
    os.makedirs(output_directory, exist_ok=True)

    print(f"Input: {config.input_path}")
    if load_path != input_path:
        print(f"Resolved images dir: {load_path}")
    print(f"Type: {input_type}")
    print(f"Output directory: {output_directory}")

    # Load input based on type
    start_load = time.time()
    image_filenames = None  # Track filenames for directory mode

    if is_video:
        frames, fps, total_frames = load_video_frames(str(load_path))
    elif is_directory:
        eval_file_names = getattr(config, "eval_image_filenames", None)
        if eval_file_names:
            base_dir = str(load_path)
            frames, image_filenames = load_images_from_list(base_dir, eval_file_names)
        else:
            frames, image_filenames = load_images_from_directory(str(load_path))
        fps = None
        total_frames = len(frames)
    else:
        frames = [load_image(str(load_path))]
        fps = None
        total_frames = 1

    load_time = time.time() - start_load
    print(f"Input loading time: {load_time:.2f} seconds\n")

    return InferenceInput(
        frames=frames,
        fps=fps,
        total_frames=total_frames,
        image_filenames=image_filenames,
        output_directory=output_directory,
        input_type=input_type,
        input_path=input_path,
        load_time=load_time
    )

def encode_frame(frame_rgb: np.ndarray) -> str:
    """
    Encode RGB frame to base64 JPEG string.

    Args:
        frame_rgb: RGB image as numpy array

    Returns:
        Base64 encoded JPEG string
    """
    # Convert RGB to BGR for OpenCV encoding
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Encode as JPEG
    success, buffer = cv2.imencode('.jpg', frame_bgr)
    if not success:
        raise ValueError("Failed to encode frame as JPEG")

    # Convert to base64
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

def send_batch_to_server(
    frames_with_ids: List[Tuple[int, np.ndarray]],
    server_url: str,
    text_prompts: List[str] = None,
    timeout: int = 120
) -> List[Dict[str, Any]]:
    """
    Send a batch of frames to the YOLO-World server for inference.

    Args:
        frames_with_ids: List of (frame_id, frame_rgb) tuples
        server_url: Server endpoint URL
        text_prompts: Optional list of class names to detect (shared across batch)
        timeout: Request timeout in seconds (default 120 for batch processing)

    Returns:
        List of detection result dictionaries
    """
    # Encode all frames
    images_data = []
    for frame_id, frame_rgb in frames_with_ids:
        img_base64 = encode_frame(frame_rgb)
        images_data.append({
            "frame_id": frame_id,
            "image": img_base64
        })

    # Prepare batch payload
    payload = {"images": images_data}

    # Add text prompts if provided
    if text_prompts is not None:
        payload["caption"] = text_prompts

    # Debug: print batch info (optional - can be removed)
    # print(f"Sending batch of {len(images_data)} images to server")

    # Send batch request to main endpoint
    try:
        response = requests.post(
            server_url,
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        return result.get("results", [])
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Batch server request failed: {e}")

def dict_to_frame_detections(result_dict: Dict[str, Any], frame_id: int) -> FrameDetections:
    """
    Convert server response dictionary to FrameDetections object.

    Args:
        result_dict: Detection results from server
        frame_id: Frame identifier

    Returns:
        FrameDetections object
    """
    detections = []
    
    # Process detections
    detections_list = result_dict.get('detections', [])
    # Handle nested detections if any
    if isinstance(detections_list, dict):
        # Could happen if server sends {detections: {...}} but usually it's list
        pass
        
    for det_dict in detections_list:
        bbox_dict = det_dict['bbox']
        bbox = BoundingBox(
            x_min=float(bbox_dict['x_min']),
            y_min=float(bbox_dict['y_min']),
            x_max=float(bbox_dict['x_max']),
            y_max=float(bbox_dict['y_max'])
        )

        detection = Detection(
            bbox=bbox,
            class_name=det_dict['class_name'],
            confidence=float(det_dict['confidence']),
            class_id=int(det_dict['class_id']) if det_dict.get('class_id') is not None else None
        )
        detections.append(detection)

    fd = FrameDetections(
        frame_id=frame_id,
        detections=detections,
        frame_width=result_dict.get('frame_width'),
        frame_height=result_dict.get('frame_height')
    )
    
    # Attach metrics if present
    if 'metrics' in result_dict:
        fd.metrics = result_dict['metrics']
        
    return fd


def get_unique_output_directory(base_directory: str, name: str) -> str:
    """
    Create a unique output directory. If directory exists, append timestamp.

    Args:
        base_directory: Base output directory
        name: Name for the subdirectory (e.g., video filename stem)

    Returns:
        Unique directory path
    """
    output_dir = os.path.join(base_directory, name)

    # If directory exists, append timestamp
    if os.path.exists(output_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_directory, f"{name}_{timestamp}")

    return output_dir


def save_inference_results(
    frames,
    all_detections: List[FrameDetections],
    output_directory: str,
    config,
    input_path: str,
    input_type: str,
    fps: Optional[float] = None,
    total_frames: int = 0,
    inference_time: float = 0.0,
    text_prompts: Optional[List[str]] = None,
    gpu_memory_summary: Optional[Dict[str, Any]] = None,
    model_size_mb: float = 0.0,
    image_filenames: Optional[List[str]] = None
):
    """
    Save inference results: annotated frames, annotated video/images, and JSON results.

    Args:
        frames: List of RGB frames
        all_detections: List of FrameDetections objects
        output_directory: Directory to save results
        config: InferenceConfig or similar config object
        input_path: Path to input video/image
        input_type: Type of input ('video', 'image', 'directory')
        fps: Frames per second (for video)
        total_frames: Total number of frames
        inference_time: Time taken for inference
        text_prompts: Text prompts used for detection
        gpu_memory_summary: GPU memory usage summary
        model_size_mb: Model size in MB
        image_filenames: Original image filenames (for directory mode)
    """
    is_video = input_type == 'video'
    is_directory = input_type == 'directory'

    # Use text prompts from parameter or fall back to config
    prompts_to_use = text_prompts if text_prompts is not None else config.text_prompts

    # Create annotator
    annotator = FrameAnnotator(
        show_confidence=config.show_confidence,
        show_class_name=config.show_class_name,
        bbox_thickness=config.bbox_thickness,
        font_scale=config.font_scale
    )

    # Save annotated video
    if config.save_annotated and is_video:
        print("\nGenerating annotated video...")
        output_video_path = os.path.join(
            output_directory,
            f"{Path(input_path).stem}_annotated.mp4"
        )

        with AnnotatedVideoWriter(output_video_path, fps=fps) as writer:
            for frame, frame_det in tqdm(
                zip(frames, all_detections),
                desc="Writing video",
                total=len(frames)
            ):
                annotated_frame = annotator.annotate_frame(frame, frame_det)
                writer.write_frame(annotated_frame)

        print(f"Annotated video saved to: {output_video_path}")

    # Save annotated single image
    if config.save_annotated and not is_video and not is_directory:
        print("\nGenerating annotated image...")
        output_image_path = os.path.join(
            output_directory,
            f"{Path(input_path).stem}_annotated.jpg"
        )

        annotated_frame = annotator.annotate_frame(frames[0], all_detections[0])

        annotated_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image_path, annotated_bgr)
        print(f"Annotated image saved to: {output_image_path}")

    # Save annotated images for directory mode
    if config.save_annotated and is_directory:
        print("\nGenerating annotated images for directory...")
        annotated_images_dir = os.path.join(output_directory, "annotated_images")
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

    # Save results JSON
    if config.save_results_json:
        print("\nSaving results to JSON...")
        output_json_path = os.path.join(output_directory, config.results_filename)

        ResultsWriter.write_results(
            output_path=output_json_path,
            model_config=config.config_file,
            model_checkpoint=config.checkpoint,
            input_path=input_path,
            all_frame_detections=all_detections,
            text_prompts=prompts_to_use,
            execution_time=inference_time,
            fps=fps,
            total_frames=total_frames,
            gpu_memory=gpu_memory_summary if gpu_memory_summary else {},
            model_name=config.model_name,
            display_name=config.display_name,
            model_size_mb=model_size_mb
        )

        ResultsWriter.print_summary(output_json_path)
