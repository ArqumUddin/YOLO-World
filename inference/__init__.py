"""
YOLO-World Inference Module

Provides inference capabilities for YOLO-World models on images and videos.
"""

from .yolo_world import YOLOWorld
from .detection_result import Detection, BoundingBox, FrameDetections
from .results_writer import ResultsWriter
from .visualization import FrameAnnotator, AnnotatedVideoWriter
from .server_model import YOLOWorldServer
from .config import ServerConfig, InferenceConfig, ClientConfig

__all__ = [
    'YOLOWorld',
    'YOLOWorldServer',
    'Detection',
    'BoundingBox',
    'FrameDetections',
    'ResultsWriter',
    'FrameAnnotator',
    'AnnotatedVideoWriter',
    'host_model',
    'str_to_image',
    'ServerConfig',
    'InferenceConfig',
    'ClientConfig'
]
