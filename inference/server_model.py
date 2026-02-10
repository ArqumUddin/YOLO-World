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
        Process incoming prediction request (single image or batch).

        Args:
            payload: Dictionary containing either:
                Single image mode:
                    - image: Base64 encoded image string
                    - caption: Optional string or list of class names to detect
                Batch mode:
                    - images: List of dicts with 'frame_id' and 'image' (base64 encoded)
                    - caption: Optional string or list of class names to detect (shared across batch)

        Returns:
            Dictionary containing detection results:
                Single image: {frame_id, frame_width, frame_height, detections, ...}
                Batch: {results: [{frame_id, ...}, {frame_id, ...}, ...]}
        """
        # Detect if this is a batch request or single image request
        if "images" in payload:
            # Batch processing mode
            return self._process_batch(payload)
        else:
            # Single image processing mode (backward compatible)
            return self._process_single(payload)

    def _process_single(self, payload: dict) -> dict:
        """Process single image request."""
        print("Processing single image")
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

    def _process_batch(self, payload: dict) -> dict:
        """Process batch request."""
        images_data = payload.get("images", [])
        caption = payload.get("caption", None)

        print(f"Processing batch of {len(images_data)} images")

        # Parse prompts once (shared across all images in batch)
        if caption is not None:
            if isinstance(caption, str):
                prompts = [[c.strip()] for c in caption.rstrip(" .").split(" . ")] + [[' ']]
            elif isinstance(caption, list):
                prompts = [[c.strip()] for c in caption] + [[' ']]
            else:
                prompts = None
        else:
            prompts = None

        # Process each image in the batch
        results = []
        for img_data in images_data:
            frame_id = img_data.get("frame_id", 0)
            img_str = img_data.get("image")

            try:
                img_np = self.str_to_image(img_str)
                predictions = self.predict(image=img_np, prompts=prompts, frame_id=frame_id)
                results.append(predictions.to_dict())
            except Exception as e:
                # Log error and return empty result for this frame
                print(f"Error processing frame {frame_id}: {e}")
                results.append({
                    "frame_id": frame_id,
                    "frame_width": 0,
                    "frame_height": 0,
                    "num_detections": 0,
                    "detections": [],
                    "error": str(e)
                })

        return {"results": results}