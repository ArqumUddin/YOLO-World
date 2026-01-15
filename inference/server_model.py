import numpy as np
from .yolo_world import YOLOWorld
import cv2
import base64

class YOLOWorldServer(YOLOWorld):
    """
    YOLO-World server that handles REST API requests.
    """
    def str_to_image(self, img_str: str) -> np.ndarray:
        """
        Convert base64 encoded image to numpy array (RGB format).

        Expects: JPEG/PNG/etc. encoded image (standard image formats)

        Client encoding examples:
        - OpenCV: cv2.imencode('.jpg', cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
        - PIL: Image.fromarray(rgb_frame).save(buffer, 'JPEG')

        Returns:
            RGB image as numpy array (H, W, 3)
        """
        img_bytes = base64.b64decode(img_str)
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise ValueError("Failed to decode image. Ensure image is JPEG/PNG encoded.")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

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
                prompts = [[c.strip()] for c in caption.rstrip(" .").split(" . ")] + [[' ']]
            elif isinstance(caption, list):
                prompts = [[c.strip()] for c in caption] + [[' ']]
            else:
                prompts = None
        else:
            prompts = None

        predictions = self.predict(image=img_np, prompts=prompts)
        return predictions.to_dict()