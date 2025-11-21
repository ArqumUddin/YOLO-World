from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class BoundingBox:
    """
    Bounding box representation.
    Uses [x_min, y_min, x_max, y_max] format (absolute coordinates).
    """
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        """Get bounding box width."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Get bounding box height."""
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        """Get bounding box area."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Get center coordinates (x, y)."""
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    def to_xywh(self) -> Tuple[float, float, float, float]:
        """Convert to (x, y, width, height) format."""
        return (self.x_min, self.y_min, self.width, self.height)

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Convert to (x_min, y_min, x_max, y_max) format."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'x_min': float(self.x_min),
            'y_min': float(self.y_min),
            'x_max': float(self.x_max),
            'y_max': float(self.y_max),
            'width': float(self.width),
            'height': float(self.height)
        }

@dataclass
class Detection:
    """
    Single object detection result.
    """
    bbox: BoundingBox
    class_name: str
    confidence: float
    class_id: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'bbox': self.bbox.to_dict(),
            'class_name': self.class_name,
            'confidence': float(self.confidence),
            'class_id': int(self.class_id) if self.class_id is not None else None
        }

@dataclass
class FrameDetections:
    """
    Detection results for a single frame.
    """
    frame_id: int
    detections: List[Detection] = field(default_factory=list)
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'frame_id': self.frame_id,
            'frame_width': self.frame_width,
            'frame_height': self.frame_height,
            'num_detections': len(self.detections),
            'detections': [det.to_dict() for det in self.detections]
        }

    def get_detection_counts(self) -> dict:
        """Get count of detections per class."""
        counts = {}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts
