"""
YOLO-World Inference server for REST APIs
"""
import numpy as np
from .yolo_world import YOLOWorld
import cv2
from flask import Flask, request, jsonify
import base64

def str_to_image(img_str: str) -> np.ndarray:
    """Convert base64 encoded string to numpy image."""
    img_bytes = base64.b64decode(img_str)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_np = cv2.imdecode(img_arr, cv2.IMREAD_ANYCOLOR)
    return img_np

def host_model(model: 'YOLOWorldServer', name: str, port: int = 5000) -> None:
    """
    Host a model as a REST API using Flask.

    Args:
        model: YOLOWorldServer instance with process_payload method
        name: Endpoint name (e.g., 'yolo_world')
        port: Port to run the server on
    """
    app = Flask(__name__)

    @app.route(f"/{name}", methods=["POST"])
    def process_request():
        payload = request.json
        return jsonify(model.process_payload(payload))

    app.run(host="localhost", port=port)

class YOLOWorldServer(YOLOWorld):
    """YOLO-World server that handles REST API requests."""

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
        img_np = str_to_image(payload["image"])
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="YOLO-World REST API Server"
    )

    parser.add_argument("--port", type=int, default=12182,
                        help="Port to run the server on (default: 12182)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YOLO-World config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to YOLO-World checkpoint file")
    parser.add_argument("--prompts", type=str, default="person,car,dog,cat,bird",
                        help="Comma-separated list of class names (default: person,car,dog,cat,bird)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu, default: auto-detect)")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--nms", type=float, default=0.7,
                        help="NMS IoU threshold (default: 0.7)")
    parser.add_argument("--max-detections", type=int, default=100,
                        help="Maximum detections per image (default: 100)")

    args = parser.parse_args()

    # Parse prompts
    text_prompts = [p.strip() for p in args.prompts.split(",")]

    print("=" * 80)
    print("YOLO-World Server Initialization")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device or 'auto-detect'}")
    print(f"Port: {args.port}")
    print(f"Prompts: {text_prompts}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"NMS threshold: {args.nms}")
    print(f"Max detections: {args.max_detections}")
    print("=" * 80)
    print("\nLoading model...")

    # Initialize server
    yolo_server = YOLOWorldServer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        text_prompts=text_prompts,
        device=args.device,
        confidence_threshold=args.confidence,
        nms_threshold=args.nms,
        max_detections=args.max_detections
    )

    print("\n" + "=" * 80)
    print(f"Model loaded!")
    print(f"Server ready! Listening on http://localhost:{args.port}/yolo_world")
    print("=" * 80)

    # Start server
    host_model(yolo_server, name="yolo_world", port=args.port)
