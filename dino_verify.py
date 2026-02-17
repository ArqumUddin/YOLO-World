import torch
import numpy as np

# --- TUNABLE PARAMETERS ---
# Verify ONLY if YOLO confidence is below this score.
# If confidence is high (>= 0.8), we trust YOLO and skip verification.
CONFIDENCE_THRESHOLD = 0.8

# Labels not in prototypes are skipped.
# (Logic handled inside verify function by checking prototypes keys)

# Threshold for 'global_adaptive_avg' mode (features > this score are averaged)
ADAPTIVE_AVG_THRESHOLD = 0.3
# --------------------------

class DinoFeatureVerifier:
    def __init__(self, correction_enabled: bool = False, verification_mode: str = "global_adaptive_avg"):
        """
        Args:
            correction_enabled: Whether to allow correcting labels to better matches. (Hardcoded False in usage)
            verification_mode: Verification strategy. (Hardcoded global_adaptive_avg in usage)
        """
        self.correction_enabled = correction_enabled
        self.verification_mode = verification_mode

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
        
        current_yolo_label = yolo_result['label']
        result_label = current_yolo_label
        corrected = False
        action = 'verified_kept'

        # Global Adaptive Average Mode Only
        # 1. Compare against ALL labels' individual features.
        # 2. For each label, average the similarity scores of ONLY the features that have sim > ADAPTIVE_AVG_THRESHOLD (0.3).
        # 3. If no features > 0.3 for a label, score is 0.0.
        # 4. Use the best score to decide (Keep/Correct/Discard).
        
        label_scores = []
        
        for label, feats in prototypes.items():
            # feats: (N, 1024), embedding: (1024,)
            sims = torch.mv(feats, embedding)
            
            # Mask for features > threshold
            good_mask = sims > ADAPTIVE_AVG_THRESHOLD
            good_sims = sims[good_mask]
            
            if good_sims.numel() > 0:
                score = good_sims.mean().item()
                # Secondary tie-breaker: max similarity found for this label
                max_sim_tiebreaker = good_sims.max().item() 
            else:
                score = 0.0
                max_sim_tiebreaker = 0.0

            label_scores.append({
                'label': label,
                'score': score,
                'max_sim': max_sim_tiebreaker
            })
        
        # Find best score
        if not label_scores:
            best_score = 0.0
            winners = []
        else:
            # Sort descending by score, then by max_sim
            label_scores.sort(key=lambda x: (x['score'], x['max_sim']), reverse=True)
            best_score = label_scores[0]['score']
            
            # Use top 3 labels
            top3_labels_scores = label_scores[:3]
            
            # Check for winners (allow ties) - still useful for logging, but not strictly "winners" in top 3 logic
            winners = [x for x in label_scores if x['score'] == best_score]
        
        best_sim = best_score
        
        if best_score == 0.0: # No good matches for ANY label
            result_label = None
            corrected = False
            action = 'discarded_no_adaptive_matches'
        else:
             # Check if YOLO label is among the Top 3
             top3_labels = [w['label'] for w in top3_labels_scores]
             
             if current_yolo_label in top3_labels:
                 # Verified (YOLO is in top 3)
                 pass
             else:
                 # YOLO label is not in Top 3
                 if self.correction_enabled:
                     best_winner = winners[0]['label']
                     result_label = best_winner
                     corrected = True
                     action = 'corrected_global_adaptive_top3'
                 else:
                     # Correction disabled -> Discard detection because YOLO label wasn't in top 3
                     result_label = None
                     corrected = False
                     action = 'discarded_correction_disabled_global_adaptive_top3'
            
        return {
            'final_label': result_label,
            'is_corrected': corrected,
            'similarity_score': best_sim,
            'action': action
        }