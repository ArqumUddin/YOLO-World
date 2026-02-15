"""
YOLO-World + DINOv2 Inference Server

This server implements a hybrid detection pipeline:
1. YOLO-World for object detection (bounding boxes + initial class)
2. DINOv2 for feature verification (O(1) texture/feature check)

It is designed to be a drop-in replacement for the standard YOLO-World server.
"""
import sys
import os
import time
import torch
import numpy as np
import concurrent.futures
from PIL import Image
from typing import List, Optional, Dict, Any
from flask import Flask, request, jsonify
import argparse
import yaml
from pathlib import Path
import cv2
import base64

# Add YOLO-World to path so we can import its modules
# sys.path.append("YOLO-World")

# Import YOLO-World modules
from inference.server_model import YOLOWorldServer
from inference.detection_result import Detection, BoundingBox

# Modular imports
from dino_precompute import DinoFeaturePrecomputer
from dino_verify import DinoFeatureVerifier

# --- CONFIG ---
YOLO_CONFIG_PATH = "configs/inference/yolo_world_v2_x.py"
YOLO_WEIGHTS_PATH = "weights/yolo_world_v2_x_stage1.pth"
REAL_REF_DIR = "DINO_test/sel_feats/real_ref"
TARGET_LABELS = ["chair", "bird", "grill", "umbrella", "hospital sign"]

class DinoYoloWorldServer(YOLOWorldServer):
    """
    Extended YOLO-World Server that adds DINOv2 verification.
    """
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        text_prompts: List[str],
        device: Optional[str] = None,
        confidence_threshold: float = 0.05,
        nms_threshold: float = 0.5,
        max_detections: int = 100,
        port: int = 5000,
        verify_enabled: bool = True,
        benchmark_mode: bool = False
    ):
        # Initialize parent YOLO-World model
        super().__init__(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            text_prompts=text_prompts,
            device=device,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            max_detections=max_detections
        )
        
        self.verify_enabled = verify_enabled
        self.benchmark_mode = benchmark_mode

        if self.verify_enabled:
            print("--- Initializing DINO Verification Modules ---")
            # Thread pool for parallel execution
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
            
            # Load DINO components
            print("Loading DINO Precomputer...")
            # Use same device as YOLO if possible, or defaulting to cuda:0
            self.dino_precomputer = DinoFeaturePrecomputer(device=self.device)
            self.dino_verifier = DinoFeatureVerifier()
            
            # Load Prototypes
            self.prototypes = self.load_prototypes()
            print("--- DINO Modules Ready ---\n")

    def process_payload(self, payload: dict) -> dict:
        """
        Process incoming prediction request (single image or batch).
        Metrics tracking added at this level.
        """
        # Detect if this is a batch request or single image request
        if "images" in payload:
            # Batch processing mode
            return self._process_batch(payload)
        else:
            # Single image processing mode (backward compatible)
           return super().process_payload(payload)
    
    def _process_batch(self, payload: dict) -> dict:
        """Process batch request with optional metrics."""
        t_batch_start = time.time()
        
        images_data = payload.get("images", [])
        caption = payload.get("caption", None)

        if self.benchmark_mode:
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
        decode_times = []
        
        for img_data in images_data:
            frame_id = img_data.get("frame_id", 0)
            img_str = img_data.get("image")

            try:
                # Measure decoding time if benchmarking
                if self.benchmark_mode:
                    t_decode_start = time.time()
                    img_np = self.str_to_image(img_str)
                    t_decode_end = time.time()
                    decode_times.append(t_decode_end - t_decode_start)
                else:
                    img_np = self.str_to_image(img_str)
                
                # Run Inference (Includes DINO + YOLO + Verify)
                predictions = self.predict(image=img_np, prompts=prompts, frame_id=frame_id)
                
                # Update metrics if in benchmark mode and metrics exist
                if self.benchmark_mode and hasattr(predictions, 'metrics') and predictions.metrics:
                    m = predictions.metrics
                    # Add decode time
                    decode_ms = (t_decode_end - t_decode_start) * 1000
                    m['decode_ms'] = round(decode_ms, 2)
                    
                    # Calculate server E2E time
                    m['server_e2e_ms'] = round(m['total_roundtrip_ms'] + decode_ms, 2)
                    
                    predictions.metrics = m
                
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

        if self.benchmark_mode:
            t_batch_total = time.time() - t_batch_start
            
            # Calculate summary metrics
            avg_decode = sum(decode_times) / len(decode_times) if decode_times else 0
            
            print(f"\n[BATCH SUMMARY] Size: {len(images_data)}")
            print(f"  Total Batch Time: {t_batch_total*1000:.2f} ms")
            print(f"  Avg Time / Image: {(t_batch_total/len(images_data))*1000:.2f} ms")
            print(f"  Avg Decode Time:  {avg_decode*1000:.2f} ms")

        return {"results": results}

    def load_prototypes(self):
        prototypes = {}
        print(f"Loading feature prototypes from {REAL_REF_DIR}...")
        if not os.path.exists(REAL_REF_DIR):
            print(f"Warning: Prototype directory {REAL_REF_DIR} not found.")
            return prototypes
            
        for label in TARGET_LABELS:
            path = os.path.join(REAL_REF_DIR, f"{label}.npy")
            if os.path.exists(path):
                feats = np.load(path)
                proto = torch.from_numpy(feats).mean(dim=0).to(self.device)
                prototypes[label] = proto / proto.norm()
            else:
                print(f"Warning: No features found for {label}")
        return prototypes

    def _run_dino_timed(self, images):
        """Helper to run DINO and measure actual execution time."""
        t0 = time.time()
        result = self.dino_precomputer.compute_feature_map(images)
        t1 = time.time()
        return result, (t1 - t0)

    def predict(
        self,
        image: np.ndarray,
        prompts: List[str] = None,
        frame_id: int = 0
    ):
        """
        Overridden predict method that runs YOLO and DINO in parallel.
        """
        # If verification is disabled or no prototypes, mostly fall back to normal YOLO
        # But we still want to benefit from the structure
        if not self.verify_enabled:
            return super().predict(image, prompts, frame_id)

        # 0. Start Timer (t0)
        t_start = time.time()

        # --- PARALLEL BLOCK ---
        # 1. Start DINO Precompute in background (Thread 2)
        # Convert numpy (BGR) to PIL (RGB) for DINO
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        
        # Submit wrapped task
        dino_future = self.executor.submit(self._run_dino_timed, [pil_image])
        
        # 2. Run YOLO Detection (Thread 1 - Main)
        # We call the parent's predict which runs the standard YOLO inference
        t_yolo_start = time.time()
        detections_obj = super().predict(image, prompts, frame_id)
        t_yolo_end = time.time() 
        
        # 3. Wait for DINO to finish
        # Result is a tuple (dino_maps, duration)
        dino_maps, dino_duration = dino_future.result()
        
        dino_map = dino_maps[0] 
        # --- END PARALLEL BLOCK ---

        # 4. Verification & Correction
        t_verify_start = time.time()
        
        verified_detections = []
        
        for det in detections_obj.detections:
            # We need to map the class_name to something we might have prototypes for.
            # Currently prototypes are keyed by TARGET_LABELS.
            # If YOLO detects "hospital sign", we verify against "hospital sign" prototype.
            
            # Map YOLO detection to format expected by verifier
            # bbox in Detection is [x_min, y_min, x_max, y_max]
            bbox_list = [det.bbox.x_min, det.bbox.y_min, det.bbox.x_max, det.bbox.y_max]
            
            yolo_det_dict = {
                'label': det.class_name,
                'bbox': bbox_list,
                'conf': det.confidence
            }
            
            # CALL VERIFIER
            verify_out = self.dino_verifier.verify(dino_map, yolo_det_dict, self.prototypes)
            
            # Update detection based on verification
            # If label changed, update class_name
            # We might also update confidence if we have a way to combine them, 
            # but for now we keep YOLO confidence or maybe use similarity?
            # The verify_out structure:
            # {'final_label': str, 'is_corrected': bool, 'similarity_score': float, 'action': str}
            
            new_det = det
            if verify_out['is_corrected']:
                new_det.class_name = verify_out['final_label']
                # Ideally we'd update class_id too if we knew the ID for the new label
                # But class_id is optional and index-based.
                # If we change the label to something not in the prompt list, class_id might be invalid.
                # safely set it to None if corrected
                new_det.class_id = None
                
            verified_detections.append(new_det)

        detections_obj.detections = verified_detections
        t_verify_end = time.time() # T3: Verification finished
        
        if self.benchmark_mode:
            # Calculate Logic Latencies
            # 1) DINO Precompute Time: Actual duration returned from thread
            dino_time = dino_duration
            
            # 2) YOLO Detection Time: Main thread duration
            yolo_time = t_yolo_end - t_yolo_start
            
            # 3) Verification Time: Logic only
            verify_time = t_verify_end - t_verify_start
            
            # 4) Total Time: Wall clock from start to return
            total_time = t_verify_end - t_start

            # For benchmarking visibility, we can print here or attach to the object
            metrics_dict = {
                "dino_precompute_ms": round(dino_time * 1000, 2),
                "yolo_detection_ms": round(yolo_time * 1000, 2),
                "verification_ms": round(verify_time * 1000, 2),
                "total_roundtrip_ms": round(total_time * 1000, 2)
            }
            
            print(f"\n[BENCHMARK] Image Processing:")
            print(f"  1. DINO Precompute: {metrics_dict['dino_precompute_ms']} ms")
            print(f"  2. YOLO Detection:  {metrics_dict['yolo_detection_ms']} ms")
            print(f"  3. Verification:    {metrics_dict['verification_ms']} ms")
            print(f"  4. Total Roundtrip: {metrics_dict['total_roundtrip_ms']} ms")
            
            # Attach metrics to result object
            if hasattr(detections_obj, 'metrics'):
                detections_obj.metrics = metrics_dict

        return detections_obj

def host_model(model_server: DinoYoloWorldServer, name: str, port: int = 5000) -> None:
    """
    Host the hybrid model as a REST API using Flask.
    """
    app = Flask(__name__)

    @app.route(f"/{name}", methods=["POST"])
    def process_request():
        try:
            payload = request.json
            if not payload:
                return jsonify({"error": "No JSON payload provided"}), 400
            
            print(f"Received request... processing.")
            start_t = time.time()
            results = model_server.process_payload(payload)
            total_duration = time.time() - start_t
            print(f"Total Request Latency: {total_duration*1000:.2f} ms")
            
            return jsonify(results)
        except Exception as e:
            print(f"Error processing request: {e}")
            return jsonify({"error": str(e)}), 500

    print(f"Starting DinoYoloWorld Server on port {port}...")
    app.run(host="0.0.0.0", port=port)

def run_benchmark(model_server):
    """
    Run a local benchmark on the evaluation images without starting the server.
    """
    print("\n" + "="*40)
    print("STARTING BENCHMARK MODE")
    print("="*40)

    eval_dir = "DINO_test/small_eval"
    if not os.path.exists(eval_dir):
        print(f"Error: Evaluation directory {eval_dir} not found.")
        return

    all_files = [os.path.join(eval_dir, f) for f in os.listdir(eval_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Take first batch of 8
    BATCH_SIZE = 8
    if len(all_files) < BATCH_SIZE:
        print(f"Not enough images for a batch of {BATCH_SIZE}. Using {len(all_files)}")
        test_batch = all_files
    else:
        test_batch = all_files[:BATCH_SIZE]

    print(f"Processing batch of {len(test_batch)} images...")
    
    # Warmup
    print("Warming up...")
    dummy = cv2.imread(test_batch[0])
    model_server.predict(dummy) 
    print("Warmup done.")

    total_start = time.time()
    
    for i, img_path in enumerate(test_batch):
        print(f"\nImage {i+1}/{len(test_batch)}: {os.path.basename(img_path)}")
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        # Run prediction (Metrics are printed inside predict)
        _ = model_server.predict(image)
        
    avg_time = (time.time() - total_start) / len(test_batch)
    print("\n" + "="*40)
    print(f"Benchmark Complete.")
    print(f"Average Total Latency per Image: {avg_time*1000:.2f} ms")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO-World + DINO Server")
    
    parser.add_argument("--port", type=int, default=12182, help="Port to run the server on")
    parser.add_argument("--config", type=str, default=YOLO_CONFIG_PATH, help="Path to model config")
    parser.add_argument("--checkpoint", type=str, default=YOLO_WEIGHTS_PATH, help="Path to checkpoint")
    parser.add_argument("--prompts", type=str, default=",".join(TARGET_LABELS), help="Comma-separated initial prompts")
    parser.add_argument("--no-verify", action="store_true", help="Disable DINO verification (YOLO only mode)")
    parser.add_argument("--benchmark", action="store_true", help="Run local benchmark instead of starting server")
    parser.add_argument("--enable-metrics", action="store_true", help="Enable detailed latency metrics in server logs")
    
    args = parser.parse_args()

    # Parse prompts
    initial_prompts = [p.strip() for p in args.prompts.split(",")]
    
    # If running local benchmark tool, force metrics on
    use_metrics = args.enable_metrics or args.benchmark

    # Initialize Server
    print("Initializing Model...")
    server = DinoYoloWorldServer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        text_prompts=initial_prompts,
        port=args.port,
        verify_enabled=not args.no_verify,
        benchmark_mode=use_metrics,
        # Default thresholds
        confidence_threshold=0.05,
        nms_threshold=0.5
    )
    
    if args.benchmark:
        run_benchmark(server)
    else:
        # Host
        host_model(server, name="yolo_world", port=args.port)
