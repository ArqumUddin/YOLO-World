import torch
import numpy as np

# --- TUNABLE PARAMETERS ---
# Verify ONLY if YOLO confidence is below this score.
# If confidence is high (>= 0.8), we trust YOLO and skip verification.
CONFIDENCE_THRESHOLD = 0.8  

# Verify ONLY if YOLO predicts one of our "seen" labels (prototypes).
# Labels not in prototypes are skipped.
# (Logic handled inside verify function by checking prototypes keys)

# Correct the prediction ONLY if the best similarity score is above this threshold.
CORRECTION_THRESHOLD = 0.5 
# --------------------------

class DinoFeatureVerifier:
    def verify(self, feature_map_data, yolo_result, prototypes):
        """
        RUNS AFTER YOLO results are in (O(1) lookups).
        
        Args:
            feature_map_data: Dict from DinoFeaturePrecomputer.
                              {
                                'feature_map': tensor (1, h_grid, w_grid, 1024),
                                'original_size': (H, W),
                                'grid_size': (h_grid, w_grid)
                              }
            yolo_result: Dict containing current YOLO detection.
                         {'label': str, 'bbox': [x1, y1, x2, y2], 'conf': float}
            prototypes: Dict of reference embeddings {label: tensor(1024,)}
            
        Returns:
            dict with {
                'final_label': str,      # The label after verification (original or corrected)
                'is_corrected': bool,    # True if label changed
                'similarity_score': float, # The score of the best matching prototype
                'action': str            # Description of what happened
            }
        """
        
        # 1) Check Confidence Score
        if yolo_result['conf'] >= CONFIDENCE_THRESHOLD:
            return {
                'final_label': yolo_result['label'],
                'is_corrected': False,
                'similarity_score': 0.0,
                'action': 'skipped_high_conf'
            }

        # 2) Check if label is in our known prototypes (Seen Labels)
        if yolo_result['label'] not in prototypes:
             return {
                'final_label': yolo_result['label'],
                'is_corrected': False,
                'similarity_score': 0.0,
                'action': 'skipped_unknown_label_unseen'
            }

        # --- Proceed with embedding extraction and verification ---
        
        orig_h, orig_w = feature_map_data['original_size']
        grid_h, grid_w = feature_map_data['grid_size']
        
        # Scaling factors: Image -> Patch Grid
        scale_x = grid_w / orig_w
        scale_y = grid_h / orig_h
        
        x1, y1, x2, y2 = yolo_result['bbox']
        
        # Convert bbox to grid coordinates
        gx1 = int(x1 * scale_x)
        gy1 = int(y1 * scale_y)
        gx2 = int(x2 * scale_x)
        gy2 = int(y2 * scale_y)
        
        # Boundary checks
        gx1 = max(0, gx1)
        gy1 = max(0, gy1)
        gx2 = min(grid_w, max(gx1 + 1, gx2))
        gy2 = min(grid_h, max(gy1 + 1, gy2))
        
        # O(1) Slice extraction + Average Pooling
        # Extract features for the region of interest
        # feature_map shape: (1, h_grid, w_grid, 1024)
        roi_features = feature_map_data['feature_map'][0, gy1:gy2, gx1:gx2, :]
        
        # Pooling: Average the patch tokens within the bbox
        if roi_features.numel() == 0:
            # Fallback to single center point if bbox is too small
            cx, cy = (gx1 + gx2) // 2, (gy1 + gy2) // 2
            embedding = feature_map_data['feature_map'][0, cy, cx, :]
        else:
            embedding = roi_features.mean(dim=(0, 1))
            
        # Normalize
        embedding = embedding / embedding.norm()
        
        # Similarity Comparison (O(K) where K=Number of Prototypes)
        best_sim = -1.0
        best_label = None
        
        current_yolo_label = yolo_result['label']
        
        # We check against ALL prototypes to find the best match amongst seen labels
        for label, proto in prototypes.items():
            sim = torch.dot(embedding, proto).item()
            if sim > best_sim:
                best_sim = sim
                best_label = label
                
        # Correction Logic
        result_label = current_yolo_label
        corrected = False
        action = 'verified_kept'
        
        if best_sim > CORRECTION_THRESHOLD:
            if best_label is not None and best_label != current_yolo_label:
                result_label = best_label
                corrected = True
                action = 'corrected'
        else:
             action = 'low_similarity_kept'
            
        return {
            'final_label': result_label,
            'is_corrected': corrected,
            'similarity_score': best_sim,
            'action': action
        }
