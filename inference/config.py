import os
import json
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
    def port(self) -> Optional[int]:
        """Get server port (optional)."""
        return self.config.get('server', {}).get('port', None)

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
    def save_annotated(self) -> bool:
        """Whether to save annotated outputs (video/image/frames)."""
        return self.config['output'].get('save_annotated', True)

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
            f"  Save annotated: {self.save_annotated}",
            f"  Save JSON: {self.save_results_json}",
        ]
        return "\n".join(lines)


class ClientConfig:
    """
    Configuration class for YOLO-World client-server inference client.
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
        self._eval_enabled = False
        self._resolved_eval_annotations = None
        self._eval_image_filenames = None
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
        required_fields = ['input', 'output']

        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")

        if 'path' not in self.config['input']:
            raise ValueError("Input configuration must include 'path'")

        input_path = self.config['input']['path']
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        if 'directory' not in self.config['output']:
            raise ValueError("Output configuration must include 'directory'")

        eval_cfg = self.config.get('evaluation')
        self._eval_enabled = eval_cfg is not None
        if self._eval_enabled:
            if self.input_type != 'directory':
                raise ValueError("Evaluation requires input.type to be 'directory' with COCO images.")

            if not os.path.isdir(input_path):
                raise ValueError("Evaluation requires input.path to be a dataset root directory.")

            if isinstance(eval_cfg, dict):
                if 'coco_annotations' in eval_cfg or 'images_dir' in eval_cfg:
                    raise ValueError(
                        "Evaluation no longer accepts coco_annotations/images_dir. "
                        "Place annotations.json under input.path and remove those fields."
                    )

            ann_path = self._find_annotations_file(self.input_path)
            if not ann_path:
                raise ValueError(
                    "Evaluation enabled but annotations.json not found under input.path."
                )
            if not os.path.exists(ann_path):
                raise FileNotFoundError(f"COCO annotations not found: {ann_path}")

            try:
                with open(ann_path, 'r') as f:
                    coco_data = json.load(f)
                if not all(k in coco_data for k in ('images', 'annotations', 'categories')):
                    raise ValueError("COCO annotations missing required keys: images/annotations/categories")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid COCO annotations JSON: {e}") from e

            image_entries = coco_data.get('images', [])
            file_names = [img.get('file_name') for img in image_entries if img.get('file_name')]
            if not file_names:
                raise ValueError("COCO annotations contain no image file_name entries.")
            self._eval_image_filenames = file_names
            self._resolved_eval_annotations = ann_path
        else:
            # Validate that if it's a directory, it contains image files
            if os.path.isdir(input_path):
                if not self._directory_has_images(input_path):
                    raise ValueError(f"Directory contains no image files: {input_path}")

    def _find_annotations_file(self, base_dir: str) -> Optional[str]:
        """Search for annotations.json under base_dir."""
        matches = []
        for root, _dirs, files in os.walk(base_dir):
            if 'annotations.json' in files:
                matches.append(os.path.join(root, 'annotations.json'))
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        matches_sorted = sorted(matches)
        example = ", ".join(matches_sorted[:5])
        raise ValueError(
            f"Multiple annotations.json files found under input.path: {example}"
        )

    def _directory_has_images(self, directory: str) -> bool:
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        return any(
            os.path.splitext(f)[1].lower() in image_extensions
            for f in os.listdir(directory)
        )

    @property
    def input_path(self) -> str:
        """Get input path (video or image)."""
        return self.config['input']['path']

    @property
    def eval_image_filenames(self) -> Optional[List[str]]:
        """Image filenames from COCO annotations (if evaluation enabled)."""
        return self._eval_image_filenames

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
    def save_annotated(self) -> bool:
        """Whether to save annotated outputs (video/image/frames)."""
        return self.config['output'].get('save_annotated', True)

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

    @property
    def server_url(self) -> str:
        """Get server URL for client requests."""
        return self.config.get('client', {}).get('server_url', 'http://localhost:12182/yolo_world')

    @property
    def eval_coco_annotations(self) -> Optional[str]:
        """Resolved COCO annotations JSON path."""
        return self._resolved_eval_annotations

    @property
    def eval_iou_thresholds(self) -> List[float]:
        """IoU thresholds for per-image metrics."""
        if not self._eval_enabled:
            return [0.5]
        return self.config.get('evaluation', {}).get('iou_thresholds', [0.5])

    @property
    def eval_min_score(self) -> Optional[float]:
        """Minimum score for per-image metrics (None means no filtering)."""
        if not self._eval_enabled:
            return None
        return self.config.get('evaluation', {}).get('min_score')

    @property
    def eval_output_filename(self) -> str:
        """Output filename for evaluation results."""
        if not self._eval_enabled:
            return 'evaluation.json'
        return self.config.get('evaluation', {}).get('output_filename', 'evaluation.json')

    @property
    def eval_per_image_metrics(self) -> bool:
        """Whether to compute per-image metrics for evaluation."""
        if not self._eval_enabled:
            return True
        return self.config.get('evaluation', {}).get('per_image_metrics', True)

    @property
    def eval_coco_verbose(self) -> bool:
        """Whether to print COCOeval verbose output."""
        if not self._eval_enabled:
            return True
        return self.config.get('evaluation', {}).get('coco_verbose', True)

    @property
    def config_file(self) -> Optional[str]:
        """Model config file (not available on client)."""
        return None

    @property
    def checkpoint(self) -> Optional[str]:
        """Model checkpoint (not available on client)."""
        return None

    @property
    def text_prompts(self) -> List[str]:
        """Text prompts (client sends only via CLI)."""
        return []

    @property
    def model_name(self) -> str:
        """Model name/identifier."""
        return self.config.get('client', {}).get('model_name', 'yolo_world')

    @property
    def display_name(self) -> Optional[str]:
        """Display name for the model (optional)."""
        return self.config.get('client', {}).get('display_name', None)

    def __repr__(self) -> str:
        return f"ClientConfig(config_path='{self.config_path}')"

    def __str__(self) -> str:
        """Print configuration."""
        lines = [
            "=== YOLO-World Client Configuration ===",
            f"Config file: {self.config_path}",
            "",
            "Input:",
            f"  Path: {self.input_path}",
            f"  Type: {self.input_type}",
            "",
            "Output:",
            f"  Directory: {self.output_directory}",
            f"  Save annotated: {self.save_annotated}",
            f"  Save JSON: {self.save_results_json}",
            "",
            "Client:",
            f"  Server URL: {self.server_url}",
        ]
        if self._eval_enabled:
            lines.extend([
                "",
                "Evaluation:",
                f"  COCO annotations: {self.eval_coco_annotations}",
                f"  Dataset root: {self.input_path}",
                f"  IoU thresholds: {self.eval_iou_thresholds}",
                f"  Min score: {self.eval_min_score}",
                f"  Per-image metrics: {self.eval_per_image_metrics}",
            ])
        return "\n".join(lines)
