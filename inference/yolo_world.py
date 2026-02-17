"""
YOLO-World model wrapper for inference.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
from mmyolo.registry import RUNNERS
from torchvision.ops import nms
from yolo_world.models import *

from .detection_result import Detection, BoundingBox, FrameDetections

class YOLOWorld:
    """
    Wrapper for YOLO-World model to provide a clean inference interface.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        text_prompts: List[str],
        device: Optional[str] = None,
        confidence_threshold: float = 0.05,
        nms_threshold: float = 0.7,
        max_detections: int = 100,
        use_amp: bool = False
    ):
        """
        Initialize YOLO-World model.

        Args:
            config_path: Path to model config file
            checkpoint_path: Path to model checkpoint
            text_prompts: List of class names to detect (e.g., ['person', 'car', 'dog'])
            device: Device for inference ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence score for detections
            nms_threshold: NMS IoU threshold
            max_detections: Maximum number of detections per frame
            use_amp: Use automatic mixed precision
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.text_prompts = text_prompts
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.use_amp = use_amp

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Loading YOLO-World model: {config_path}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Device: {self.device}")
        print(f"Text prompts ({len(text_prompts)} classes): {text_prompts}")

        # Load config — if a YAML wrapper is passed, resolve the inner .py config
        if config_path.endswith(('.yaml', '.yml')):
            import yaml as _yaml
            with open(config_path) as f:
                wrapper = _yaml.safe_load(f)
            config_path = wrapper['model']['config_file']
            self.config_path = config_path

        self.cfg = Config.fromfile(config_path)
        self.cfg.work_dir = './work_dirs'
        self.cfg.load_from = checkpoint_path

        self.model = init_detector(self.cfg, checkpoint=checkpoint_path, device=self.device)

        if hasattr(self.cfg, 'test_pipeline'):
            test_pipeline_cfg = self.cfg.test_pipeline
        else:
            test_pipeline_cfg = get_test_pipeline_cfg(cfg=self.cfg)
        test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
        self.test_pipeline = Compose(test_pipeline_cfg)

        self.texts = [[prompt.strip()] for prompt in text_prompts] + [[' ']]

        print(f"YOLO-World model loaded successfully!")

    def predict(
        self,
        image: np.ndarray,
        prompts: List[str] = None,
        frame_id: int = 0
    ) -> FrameDetections:
        """
        Run inference on a single frame.

        Args:
            image: RGB image as numpy array (H, W, 3)
            frame_id: Frame identifier

        Returns:
            FrameDetections object with detection results
        """
        if prompts is None:
            prompts = self.texts

        height, width = image.shape[:2]

        data_info = dict(img=image, img_id=frame_id, texts=prompts)
        data_info = self.test_pipeline(data_info)

        data_batch = dict(
            inputs=data_info['inputs'].unsqueeze(0),
            data_samples=[data_info['data_samples']]
        )

        with autocast(enabled=self.use_amp), torch.no_grad():
            output = self.model.test_step(data_batch)[0]
            pred_instances = output.pred_instances

        if len(pred_instances.scores) > 0:
            keep = nms(
                pred_instances.bboxes,
                pred_instances.scores,
                iou_threshold=self.nms_threshold
            )
            pred_instances = pred_instances[keep]

        pred_instances = pred_instances[
            pred_instances.scores.float() > self.confidence_threshold
        ]

        if len(pred_instances.scores) > self.max_detections:
            indices = pred_instances.scores.float().topk(self.max_detections)[1]
            pred_instances = pred_instances[indices]

        pred_instances = pred_instances.cpu().numpy()
        detections = []
        if 'bboxes' in pred_instances and len(pred_instances['bboxes']) > 0:
            for bbox_coords, label_id, score in zip(
                pred_instances['bboxes'],
                pred_instances['labels'],
                pred_instances['scores']
            ):
                class_name = prompts[int(label_id)][0]

                bbox = BoundingBox(
                    x_min=float(bbox_coords[0]),
                    y_min=float(bbox_coords[1]),
                    x_max=float(bbox_coords[2]),
                    y_max=float(bbox_coords[3])
                )

                detection = Detection(
                    bbox=bbox,
                    class_name=class_name,
                    confidence=float(score),
                    class_id=int(label_id)
                )
                detections.append(detection)

        frame_detections = FrameDetections(
            frame_id=frame_id,
            detections=detections,
            frame_width=width,
            frame_height=height
        )

        return frame_detections

    def predict_frames(
        self,
        images: List[np.ndarray],
        start_frame_id: int = 0,
        show_progress: bool = True
    ) -> List[FrameDetections]:
        """
        Run inference on multiple frames.

        Args:
            images: List of RGB images as numpy arrays
            start_frame_id: Starting frame ID
            show_progress: Show progress bar

        Returns:
            List of FrameDetections objects
        """
        all_detections = []

        if show_progress:
            from tqdm import tqdm
            images = tqdm(images, desc="Processing frames")

        for i, image in enumerate(images):
            frame_id = start_frame_id + i
            frame_detections = self.predict(image=image, frame_id=frame_id)
            all_detections.append(frame_detections)

        return all_detections

    def get_class_names(self) -> List[str]:
        """Get list of class names the model can detect."""
        return self.text_prompts

    def get_model_size_mb(self) -> float:
        """
        Get model size in MB.

        Returns:
            Model size in megabytes
        """
        try:
            num_params = sum(p.numel() for p in self.model.parameters())
            size_mb = (num_params * 4) / (1024 ** 2)
            return float(size_mb)
        except Exception as e:
            print(f"Warning: Could not calculate model size: {e}")
            return 0.0

    def __repr__(self) -> str:
        return (
            f"YOLOWorldModel(\n"
            f"  config={self.config_path},\n"
            f"  checkpoint={self.checkpoint_path},\n"
            f"  device={self.device},\n"
            f"  classes={len(self.text_prompts)},\n"
            f"  confidence_threshold={self.confidence_threshold},\n"
            f"  nms_threshold={self.nms_threshold}\n"
            f")"
        )
