import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from .detection_result import FrameDetections


class ResultsWriter:
    """
    Write inference results to JSON format.
    """

    @staticmethod
    def write_results(
        output_path: str,
        model_config: str,
        model_checkpoint: str,
        input_path: str,
        all_frame_detections: List[FrameDetections],
        text_prompts: List[str],
        execution_time: float,
        fps: Optional[float] = None,
        total_frames: Optional[int] = None,
        gpu_memory: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        display_name: Optional[str] = None,
        model_size_mb: Optional[float] = None
    ) -> None:
        """
        Write inference results to JSON file.

        Args:
            output_path: Path to output JSON file
            model_config: Path to model config file
            model_checkpoint: Path to model checkpoint
            input_path: Path to input video or image
            all_frame_detections: List of FrameDetections objects
            text_prompts: List of text prompts used for detection
            execution_time: Total execution time in seconds
            fps: Video FPS (for video inputs)
            total_frames: Total number of frames processed
            gpu_memory: GPU memory statistics (optional)
            model_name: Model name/identifier (optional)
            display_name: Display name for the model (optional)
            model_size_mb: Model size in MB (optional)
        """
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        total_frames_processed = len(all_frame_detections)
        total_detections = sum(len(fd.detections) for fd in all_frame_detections)
        frames_with_detections = sum(1 for fd in all_frame_detections if len(fd.detections) > 0)

        detection_counts = {}
        for frame_det in all_frame_detections:
            for det in frame_det.detections:
                detection_counts[det.class_name] = detection_counts.get(det.class_name, 0) + 1

        results = {
            'metadata': {
                'model_name': model_name if model_name else 'yolo_world',
                'display_name': display_name if display_name else (model_name if model_name else 'YOLO-World'),
                'model_config': model_config,
                'model_checkpoint': model_checkpoint,
                'input_path': input_path,
                'text_prompts': text_prompts,
                'timestamp': datetime.now().isoformat(),
                'inference_type': 'video' if fps is not None else 'image'
            },
            'execution': {
                'total_time_seconds': round(execution_time, 2),
                'time_per_frame_seconds': round(
                    execution_time / total_frames_processed, 4
                ) if total_frames_processed > 0 else 0,
                'fps': fps
            },
            'basic_metrics': {
                'total_frames': total_frames_processed,
                'total_detections': total_detections,
                'frames_with_detections': frames_with_detections,
                'detection_rate': round(
                    frames_with_detections / total_frames_processed, 4
                ) if total_frames_processed > 0 else 0,
                'average_detections_per_frame': round(
                    total_detections / total_frames_processed, 2
                ) if total_frames_processed > 0 else 0,
                'detection_counts': detection_counts,
                'inference_fps': round(
                    total_frames_processed / execution_time, 2
                ) if execution_time > 0 else 0
            },
            'frames': [fd.to_dict() for fd in all_frame_detections]
        }

        if model_size_mb is not None:
            results['metadata']['model_size_mb'] = round(model_size_mb, 2)

        if gpu_memory:
            results['gpu_memory'] = gpu_memory

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    @staticmethod
    def print_summary(results_path: str):
        """
        Print summary of results from JSON file.

        Args:
            results_path: Path to results JSON file
        """
        if not os.path.exists(results_path):
            print(f"Error: Results file not found: {results_path}")
            return

        with open(results_path, 'r') as f:
            results = json.load(f)

        metadata = results['metadata']
        print(f"\nModel: {metadata.get('model_name', 'Unknown')}")
        if 'display_name' in metadata:
            print(f"Display Name: {metadata['display_name']}")
        if 'model_size_mb' in metadata:
            print(f"Model Size: {metadata['model_size_mb']:.2f} MB")
        print(f"Model Config: {metadata['model_config']}")
        print(f"Model Checkpoint: {metadata['model_checkpoint']}")
        print(f"Input: {metadata['input_path']}")
        print(f"Text Prompts: {', '.join(metadata['text_prompts'])}")
        print(f"Timestamp: {metadata['timestamp']}")
        print(f"Type: {metadata['inference_type']}")

        execution = results['execution']
        print(f"\nExecution Time: {execution['total_time_seconds']:.2f} seconds")
        print(f"Time per Frame: {execution['time_per_frame_seconds']:.4f} seconds")
        if execution.get('fps'):
            print(f"Video FPS: {execution['fps']:.2f}")

        basic_metrics = results['basic_metrics']
        print(f"\nTotal Frames: {basic_metrics['total_frames']}")
        print(f"Frames with Detections: {basic_metrics['frames_with_detections']}")
        print(f"Detection Rate: {basic_metrics['detection_rate']:.2%}")
        print(f"Average Detections per Frame: {basic_metrics['average_detections_per_frame']:.2f}")
        print(f"Total Detections: {basic_metrics['total_detections']}")
        if 'inference_fps' in basic_metrics:
            print(f"Inference FPS: {basic_metrics['inference_fps']:.2f}")

        print("\nDetections by Class:")
        for class_name, count in sorted(
            basic_metrics['detection_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"  {class_name}: {count}")

        if 'gpu_memory' in results:
            gpu_mem = results['gpu_memory']
            print(f"\nGPU Memory:")
            print(f"  Device: {gpu_mem.get('device_name', 'Unknown')}")
            if gpu_mem.get('cuda_available'):
                print(f"  Total Memory: {gpu_mem.get('total_memory_mb', 0):.2f} MB")
                print(f"  Peak Memory: {gpu_mem.get('peak_memory_mb', 0):.2f} MB")
                print(f"  Average Memory: {gpu_mem.get('average_memory_mb', 0):.2f} MB")
                print(f"  Min Memory: {gpu_mem.get('min_memory_mb', 0):.2f} MB")