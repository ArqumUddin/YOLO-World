import numpy as np
from .yolo_world import YOLOWorld
import cv2
import base64

class YOLOWorldServer(YOLOWorld):
    """
    YOLO-World server that handles REST API requests.
    """
    def str_to_image(self, img_str: str) -> np.ndarray:
        """Convert base64 encoded string to numpy image."""
        img_bytes = base64.b64decode(img_str)
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img_np = cv2.imdecode(img_arr, cv2.IMREAD_ANYCOLOR)
        return img_np

    def process_payload(self, payload: dict) -> dict:
        """
        Process incoming prediction request.

        Args:
            payload: Dictionary containing:
                - image: Base64 encoded image string
                - caption: Optional string or list of class names to detect

        Returns:
            Dictionary containing detection results
        """
        img_np = self.str_to_image(payload["image"])
        caption = payload.get("caption", None)

        if caption is not None:
            if isinstance(caption, str):
                prompts = [[c.strip()] for c in caption.rstrip(" .").split(" . ")]
            elif isinstance(caption, list):
                prompts = [[c.strip()] for c in caption]
            else:
                prompts = None
        else:
            prompts = None

        predictions = self.predict(image=img_np, prompts=prompts)
        return predictions.to_dict()