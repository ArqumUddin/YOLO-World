import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import sys

sys.path.append('..')
from detection_result import Detection, FrameDetections


class FrameAnnotator:
    """
    Annotate frames with bounding boxes and class labels.
    """

    def __init__(
        self,
        bbox_color: Tuple[int, int, int] = (0, 255, 0),
        bbox_thickness: int = 2,
        font_scale: float = 0.5,
        font_thickness: int = 1,
        show_confidence: bool = True,
        show_class_name: bool = True,
        class_colors: Optional[Dict[str, Tuple[int, int, int]]] = None
    ):
        """
        Initialize frame annotator.

        Args:
            bbox_color: Default BGR color for bounding boxes (green)
            bbox_thickness: Thickness of bounding box lines
            font_scale: Font scale for labels
            font_thickness: Font thickness for labels
            show_confidence: Show confidence scores in labels
            show_class_name: Show class names in labels
            class_colors: Optional dict mapping class names to specific colors
        """
        self.bbox_color = bbox_color
        self.bbox_thickness = bbox_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.show_confidence = show_confidence
        self.show_class_name = show_class_name
        self.class_colors = class_colors or {}

        self.color_cache = {}

    def _get_color_for_class(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for a specific class."""
        if class_name in self.class_colors:
            return self.class_colors[class_name]

        if class_name not in self.color_cache:
            np.random.seed(hash(class_name) % 2**32)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            self.color_cache[class_name] = color

        return self.color_cache[class_name]

    def _draw_detection(
        self,
        image: np.ndarray,
        detection: Detection,
        color: Optional[Tuple[int, int, int]] = None
    ):
        """
        Draw a single detection on the image.

        Args:
            image: BGR image to draw on (modified in place)
            detection: Detection object
            color: Optional color override
        """
        if color is None:
            color = self._get_color_for_class(detection.class_name)

        bbox = detection.bbox
        pt1 = (int(bbox.x_min), int(bbox.y_min))
        pt2 = (int(bbox.x_max), int(bbox.y_max))
        cv2.rectangle(image, pt1, pt2, color, self.bbox_thickness)

        label_parts = []
        if self.show_class_name:
            label_parts.append(detection.class_name)
        if self.show_confidence:
            label_parts.append(f"{detection.confidence:.2f}")

        if label_parts:
            label = " ".join(label_parts)

            (label_width, label_height), baseline = cv2.getTextSize(
                label,
                self.font,
                self.font_scale,
                self.font_thickness
            )

            # Get image dimensions
            img_height, img_width = image.shape[:2]

            # Calculate label box height
            label_box_height = label_height + baseline + 5

            # Try to place label above the bounding box
            # If it goes outside the frame, place it below or inside the box
            if int(bbox.y_min) - label_box_height >= 0:
                # Place above the box
                label_pt1 = (int(bbox.x_min), int(bbox.y_min) - label_box_height)
                label_pt2 = (int(bbox.x_min) + label_width + 5, int(bbox.y_min))
                text_pt = (int(bbox.x_min) + 2, int(bbox.y_min) - baseline - 2)
            elif int(bbox.y_max) + label_box_height <= img_height:
                # Place below the box
                label_pt1 = (int(bbox.x_min), int(bbox.y_max))
                label_pt2 = (int(bbox.x_min) + label_width + 5, int(bbox.y_max) + label_box_height)
                text_pt = (int(bbox.x_min) + 2, int(bbox.y_max) + label_height)
            else:
                # Place inside the box at the top
                label_pt1 = (int(bbox.x_min), int(bbox.y_min))
                label_pt2 = (int(bbox.x_min) + label_width + 5, int(bbox.y_min) + label_box_height)
                text_pt = (int(bbox.x_min) + 2, int(bbox.y_min) + label_height)

            # Clamp label to image width
            label_pt1 = (max(0, label_pt1[0]), max(0, label_pt1[1]))
            label_pt2 = (min(img_width, label_pt2[0]), min(img_height, label_pt2[1]))
            text_pt = (max(2, text_pt[0]), max(label_height, text_pt[1]))

            cv2.rectangle(image, label_pt1, label_pt2, color, -1)

            cv2.putText(
                image,
                label,
                text_pt,
                self.font,
                self.font_scale,
                (255, 255, 255),
                self.font_thickness,
                cv2.LINE_AA
            )

    def annotate_frame(
        self,
        image: np.ndarray,
        frame_detections: FrameDetections
    ) -> np.ndarray:
        """
        Annotate a frame with bounding boxes and labels.

        Args:
            image: Original image (RGB numpy array)
            frame_detections: FrameDetections object with detections

        Returns:
            Annotated image (RGB numpy array)
        """

        annotated = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        for detection in frame_detections.detections:
            self._draw_detection(annotated, detection)

        info_text = f"Frame {frame_detections.frame_id}"
        cv2.putText(
            annotated,
            info_text,
            (10, 30),
            self.font,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        return annotated

    def annotate_frames(
        self,
        images: List[np.ndarray],
        all_detections: List[FrameDetections]
    ) -> List[np.ndarray]:
        """
        Annotate multiple frames.

        Args:
            images: List of RGB images
            all_detections: List of FrameDetections objects

        Returns:
            List of annotated RGB images
        """
        assert len(images) == len(all_detections), \
            "Number of images and detections must match"

        annotated_frames = []
        for image, frame_detections in zip(images, all_detections):
            annotated = self.annotate_frame(image, frame_detections)
            annotated_frames.append(annotated)

        return annotated_frames
