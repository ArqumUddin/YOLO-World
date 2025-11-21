"""
YOLO-World Inference Module

Provides inference capabilities for YOLO-World models on images and videos.
"""

from .yolo_world import YOLOWorld
from .detection_result import Detection, BoundingBox, FrameDetections
from .results_writer import ResultsWriter
from .visualization import FrameAnnotator, AnnotatedVideoWriter
from .server import YOLOWorldServer, host_model, str_to_image

__all__ = [
    'YOLOWorld',
    'Detection',
    'BoundingBox',
    'FrameDetections',
    'ResultsWriter',
    'FrameAnnotator',
    'AnnotatedVideoWriter',
    'YOLOWorldServer',
    'host_model',
    'str_to_image'
]
