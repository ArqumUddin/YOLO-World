import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path

class ServerConfig:
    """
    Configuration class for YOLO-World Server.
    Parses and validates YAML configuration files.
    """
    def __init__(self, config_path: str):
        """
        Initialize configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"Empty configuration file: {self.config_path}")

        return config

    def _validate_config(self):
        """Validate required configuration parameters."""
        required_fields = ['model']

        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")

        model_cfg = self.config['model']
        if 'config_file' not in model_cfg:
            raise ValueError("Model configuration must include 'config_file'")
        if 'checkpoint' not in model_cfg:
            raise ValueError("Model configuration must include 'checkpoint'")
        if 'text_prompts' not in model_cfg:
            raise ValueError("Model configuration must include 'text_prompts'")

    @property
    def model_name(self) -> str:
        """Get model name/identifier."""
        return self.config['model'].get('name', 'yolo_world')

    @property
    def display_name(self) -> Optional[str]:
        """Get display name for the model (optional)."""
        return self.config['model'].get('display_name', None)

    @property
    def config_file(self) -> str:
        """Get model config file path."""
        return self.config['model']['config_file']

    @property
    def checkpoint(self) -> str:
        """Get model checkpoint path."""
        return self.config['model']['checkpoint']

    @property
    def text_prompts(self) -> List[str]:
        """Get text prompts list for detection."""
        return self.config['model']['text_prompts']

    @property
    def confidence_threshold(self) -> float:
        """Get confidence threshold for detections."""
        return self.config['model'].get('confidence_threshold', 0.05)

    @property
    def nms_threshold(self) -> float:
        """Get NMS threshold."""
        return self.config['model'].get('nms_threshold', 0.7)

    @property
    def max_detections(self) -> int:
        """Get maximum detections per frame."""
        return self.config['model'].get('max_detections', 100)

    @property
    def device(self) -> Optional[str]:
        """Get device for model inference (cpu/cuda)."""
        return self.config['model'].get('device', None)

    @property
    def bbox_thickness(self) -> int:
        """Get bounding box thickness."""
        return self.config.get('visualization', {}).get('bbox_thickness', 2)

    @property
    def font_scale(self) -> float:
        """Get font scale for labels."""
        return self.config.get('visualization', {}).get('font_scale', 0.5)

    @property
    def show_confidence(self) -> bool:
        """Whether to show confidence scores."""
        return self.config.get('visualization', {}).get('show_confidence', True)

    @property
    def show_class_name(self) -> bool:
        """Whether to show class names."""
        return self.config.get('visualization', {}).get('show_class_name', True)

    def __repr__(self) -> str:
        return f"ServerConfig(config_path='{self.config_path}')"

    def __str__(self) -> str:
        """Print configuration."""
        lines = [
            "=== YOLO-World Inference Configuration ===",
            f"Config file: {self.config_path}",
            "",
            "Model:",
            f"  Name: {self.model_name}",
            f"  Config: {self.config_file}",
            f"  Checkpoint: {self.checkpoint}",
            f"  Text prompts: {len(self.text_prompts)} classes",
            f"  Confidence threshold: {self.confidence_threshold}",
            f"  NMS threshold: {self.nms_threshold}",
            f"  Max detections: {self.max_detections}",
        ]
        return "\n".join(lines)


class InferenceConfig:
    """
    Configuration class for YOLO-World inference.
    Parses and validates YAML configuration files.
    """
    def __init__(self, config_path: str):
        """
        Initialize configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"Empty configuration file: {self.config_path}")

        return config

    def _validate_config(self):
        """Validate required configuration parameters."""
        required_fields = ['model', 'input', 'output']

        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")

        model_cfg = self.config['model']
        if 'config_file' not in model_cfg:
            raise ValueError("Model configuration must include 'config_file'")
        if 'checkpoint' not in model_cfg:
            raise ValueError("Model configuration must include 'checkpoint'")
        if 'text_prompts' not in model_cfg:
            raise ValueError("Model configuration must include 'text_prompts'")

        if 'path' not in self.config['input']:
            raise ValueError("Input configuration must include 'path'")

        input_path = self.config['input']['path']
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        # Validate that if it's a directory, it contains image files
        if os.path.isdir(input_path):
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            image_files = [f for f in os.listdir(input_path)
                          if os.path.splitext(f)[1].lower() in image_extensions]
            if not image_files:
                raise ValueError(f"Directory contains no image files: {input_path}")

        if 'directory' not in self.config['output']:
            raise ValueError("Output configuration must include 'directory'")

    @property
    def model_name(self) -> str:
        """Get model name/identifier."""
        return self.config['model'].get('name', 'yolo_world')

    @property
    def display_name(self) -> Optional[str]:
        """Get display name for the model (optional)."""
        return self.config['model'].get('display_name', None)

    @property
    def config_file(self) -> str:
        """Get model config file path."""
        return self.config['model']['config_file']

    @property
    def checkpoint(self) -> str:
        """Get model checkpoint path."""
        return self.config['model']['checkpoint']

    @property
    def text_prompts(self) -> List[str]:
        """Get text prompts list for detection."""
        return self.config['model']['text_prompts']

    @property
    def confidence_threshold(self) -> float:
        """Get confidence threshold for detections."""
        return self.config['model'].get('confidence_threshold', 0.05)

    @property
    def nms_threshold(self) -> float:
        """Get NMS threshold."""
        return self.config['model'].get('nms_threshold', 0.7)

    @property
    def max_detections(self) -> int:
        """Get maximum detections per frame."""
        return self.config['model'].get('max_detections', 100)

    @property
    def device(self) -> Optional[str]:
        """Get device for model inference (cpu/cuda)."""
        return self.config['model'].get('device', None)

    @property
    def input_path(self) -> str:
        """Get input path (video or image)."""
        return self.config['input']['path']

    @property
    def input_type(self) -> str:
        """
        Get input type (video, image, or directory).
        Auto-detected if not specified.
        """
        if 'type' in self.config['input']:
            return self.config['input']['type']

        path = Path(self.input_path)
        if path.is_dir():
            return 'directory'
        elif path.is_file():
            ext = path.suffix.lower()
            if ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']:
                return 'video'
            elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                return 'image'

        raise ValueError(f"Cannot determine input type for: {self.input_path}")

    @property
    def output_directory(self) -> str:
        """Get output directory."""
        return self.config['output']['directory']

    @property
    def save_annotated_video(self) -> bool:
        """Whether to save annotated video/image."""
        return self.config['output'].get('save_annotated', True)

    @property
    def save_annotated_frames(self) -> bool:
        """Whether to save individual annotated frames."""
        return self.config['output'].get('save_annotated_frames', True)

    @property
    def annotated_frames_dir(self) -> str:
        """Get directory for annotated frames."""
        frames_dir = self.config['output'].get('annotated_frames_dir', 'annotated_frames')
        return os.path.join(self.output_directory, frames_dir)

    @property
    def save_results_json(self) -> bool:
        """Whether to save results JSON."""
        return self.config['output'].get('save_json', True)

    @property
    def results_filename(self) -> str:
        """Get results JSON filename."""
        return self.config['output'].get('results_filename', 'results.json')

    @property
    def bbox_thickness(self) -> int:
        """Get bounding box thickness."""
        return self.config.get('visualization', {}).get('bbox_thickness', 2)

    @property
    def font_scale(self) -> float:
        """Get font scale for labels."""
        return self.config.get('visualization', {}).get('font_scale', 0.5)

    @property
    def show_confidence(self) -> bool:
        """Whether to show confidence scores."""
        return self.config.get('visualization', {}).get('show_confidence', True)

    @property
    def show_class_name(self) -> bool:
        """Whether to show class names."""
        return self.config.get('visualization', {}).get('show_class_name', True)

    def __repr__(self) -> str:
        return f"InferenceConfig(config_path='{self.config_path}')"

    def __str__(self) -> str:
        """Print configuration."""
        lines = [
            "=== YOLO-World Inference Configuration ===",
            f"Config file: {self.config_path}",
            "",
            "Model:",
            f"  Name: {self.model_name}",
            f"  Config: {self.config_file}",
            f"  Checkpoint: {self.checkpoint}",
            f"  Text prompts: {len(self.text_prompts)} classes",
            f"  Confidence threshold: {self.confidence_threshold}",
            f"  NMS threshold: {self.nms_threshold}",
            f"  Max detections: {self.max_detections}",
            "",
            "Input:",
            f"  Path: {self.input_path}",
            f"  Type: {self.input_type}",
            "",
            "Output:",
            f"  Directory: {self.output_directory}",
            f"  Save annotated: {self.save_annotated_video}",
            f"  Save JSON: {self.save_results_json}",
        ]
        return "\n".join(lines)
