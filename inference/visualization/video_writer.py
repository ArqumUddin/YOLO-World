"""
Video writer for creating annotated videos from frames.
"""

import cv2
import os
from typing import List
import numpy as np


class AnnotatedVideoWriter:
    """
    Create annotated videos from individual frames.
    """

    def __init__(
        self,
        output_path: str,
        fps: float = 30.0,
        codec: str = 'mp4v'
    ):
        """
        Initialize video writer.

        Args:
            output_path: Path to output video file
            fps: Frames per second for output video
            codec: Video codec (default: 'mp4v')
        """
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self.writer = None
        self.frame_size = None
        self.frame_count = 0

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def write_frame(self, frame: np.ndarray):
        """
        Write a single frame to video.

        Args:
            frame: Frame as RGB numpy array
        """
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.writer is None:
            self.frame_size = (frame_bgr.shape[1], frame_bgr.shape[0])
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.fps,
                self.frame_size
            )


        if (frame_bgr.shape[1], frame_bgr.shape[0]) != self.frame_size:
            frame_bgr = cv2.resize(frame_bgr, self.frame_size)

        self.writer.write(frame_bgr)
        self.frame_count += 1

    def write_frames(self, frames: List[np.ndarray]):
        """
        Write multiple frames to video.

        Args:
            frames: List of frames as RGB numpy arrays
        """
        for frame in frames:
            self.write_frame(frame)

    def release(self):
        """Release video writer and close file."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f"Video saved to: {self.output_path}")
            print(f"Total frames written: {self.frame_count}")

            if os.path.exists(self.output_path):
                file_size = os.path.getsize(self.output_path)
                if file_size > 0:
                    print(f"Video file size: {file_size / (1024*1024):.2f} MB")
                else:
                    print("Warning: Video file is empty (0 bytes)")
            else:
                print("Warning: Video file was not created")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
