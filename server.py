"""
YOLO-World Inference server for REST APIs

Supports both YAML config files and command-line arguments.
"""
from inference.server_model import YOLOWorldServer
from inference.config import ServerConfig
from flask import Flask, request, jsonify
import argparse
import yaml
from pathlib import Path

def host_model(model: YOLOWorldServer, name: str, port: int = 5000) -> None:
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

    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLO-World REST API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using YAML config file
  python server.py --yaml-config configs/inference/client_server/server.yaml

  # Using command-line arguments
  python server.py \\
      --config configs/pretrain/yolo_world_v2_l.py \\
      --checkpoint weights/yolo_world_v2_l_stage1.pth \\
      --prompts "person,car,dog"
        """
    )

    # Add mutually exclusive group for config methods
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--yaml-config", type=str,
                        help="Path to YAML config file (e.g., configs/inference/*.yaml)")
    config_group.add_argument("--config", type=str,
                        help="Path to YOLO-World model config file (.py)")

    # Optional arguments for command-line mode
    parser.add_argument("--checkpoint", type=str,
                        help="Path to YOLO-World checkpoint file")
    parser.add_argument("--prompts", type=str,
                        help="Comma-separated list of class names")
    parser.add_argument("--device", type=str,
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--confidence", type=float,
                        help="Confidence threshold")
    parser.add_argument("--nms", type=float,
                        help="NMS IoU threshold")
    parser.add_argument("--max-detections", type=int,
                        help="Maximum detections per image")

    # Server options
    parser.add_argument("--port", type=int, default=12182,
                        help="Port to run the server on (default: 12182)")

    args = parser.parse_args()

    # Load configuration
    if args.yaml_config:
        # Load from YAML file
        yaml_path = Path(args.yaml_config)
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML config not found: {args.yaml_config}")

        processed_server_config = ServerConfig(yaml_path)
        config_path = processed_server_config.config_file
        checkpoint_path = processed_server_config.checkpoint
        text_prompts = processed_server_config.text_prompts
        device = processed_server_config.device
        confidence = processed_server_config.confidence_threshold
        nms = processed_server_config.nms_threshold
        max_detections = processed_server_config.max_detections
        port = processed_server_config.port or args.port

        print(f"Loaded config from: {args.yaml_config}")
    else:
        # Use command-line arguments
        if not args.checkpoint or not args.prompts or args.config:
            parser.error("--checkpoint, --config, and --prompts are required when not using --yaml-config")

        config_path = args.config
        checkpoint_path = args.checkpoint
        text_prompts = [p.strip() for p in args.prompts.split(",")]
        device = args.device
        confidence = args.confidence or 0.25
        nms = args.nms or 0.7
        max_detections = args.max_detections or 100
        port = args.port

    print("=" * 80)
    print("YOLO-World Server Initialization")
    print("=" * 80)
    print(f"Model Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device or 'auto-detect'}")
    print(f"Port: {port}")
    print(f"Prompts ({len(text_prompts)}): {text_prompts}")
    print(f"Confidence threshold: {confidence}")
    print(f"NMS threshold: {nms}")
    print(f"Max detections: {max_detections}")
    print("=" * 80)
    print("\nLoading model...")

    # Initialize server
    yolo_server = YOLOWorldServer(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        text_prompts=text_prompts,
        device=device,
        confidence_threshold=confidence,
        nms_threshold=nms,
        max_detections=max_detections
    )

    print("\n" + "=" * 80)
    print(f"✓ Model loaded successfully!")
    print(f"✓ Server ready! Listening on http://0.0.0.0:{port}/yolo_world")
    print("=" * 80)
    print("\nExample request:")
    print(f"  curl -X POST http://localhost:{port}/yolo_world \\")
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"image": "<base64_encoded_image>", "caption": "person . car . dog ."}\'')
    print("=" * 80 + "\n")

    # Start server
    host_model(yolo_server, name="yolo_world", port=port)
