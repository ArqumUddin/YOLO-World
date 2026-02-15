import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

# --- TUNABLE PARAMETERS ---
DINO_PATCH_SIZE = 14
DINO_DEVICE = 'cuda:0'
# --------------------------

class DinoFeaturePrecomputer:
    def __init__(self, device=DINO_DEVICE, patch_size=DINO_PATCH_SIZE):
        self.device = device
        self.patch_size = patch_size
        print(f"Initializing DINOv2 Precomputer on {device}...")
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        self.model.to(self.device)
        self.model.eval()
        
        # Standard ImageNet normalization
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def preprocess_image(self, image):
        """
        Resizes image to be a multiple of patch_size for feature map extraction.
        Returns: tensor (1, C, H, W), list [original_H, original_W]
        """
        w, h = image.size
        
        # Resize to multiple of patch_size (14)
        # We perform a resize ensuring dimensions are multiples of 14
        new_w = (w // self.patch_size) * self.patch_size
        new_h = (h // self.patch_size) * self.patch_size
        
        # Avoid 0 dimensions
        new_w = max(new_w, self.patch_size)
        new_h = max(new_h, self.patch_size)
        
        transform = T.Compose([
            T.Resize((new_h, new_w)),
            self.normalize
        ])
        
        return transform(image).unsqueeze(0).to(self.device), (h, w)

    @torch.no_grad()
    def compute_feature_map(self, image_paths):
        """
        RUNS IN PARALLEL WITH YOLO.
        Computes the dense feature map for a list of images.
        
        Args:
            image_paths: List of file paths or PIL Images.
            
        Returns:
            list of dicts containing:
            {
                'feature_map': tensor (1, h_patches, w_patches, 1024),
                'original_size': (H, W),
                'patch_grid_size': (h_patches, w_patches)
            }
        """
        results = []
        
        # Note: For maximum speed, you might batch these tensors if images are same size.
        # Here we process individually to handle varying aspect ratios correctly without padding issues.
        for img_input in image_paths:
            if isinstance(img_input, str):
                image = Image.open(img_input).convert('RGB')
            else:
                image = img_input.convert('RGB')

            input_tensor, orig_size = self.preprocess_image(image)
            
            # Forward pass to get patch tokens
            # intermediate_layers=True not strictly needed unless using specific layers
            # forward_features returns dict with keys: 'x_norm_clstoken', 'x_norm_patchtokens'
            outputs = self.model.forward_features(input_tensor)
            
            # x_norm_patchtokens: (B, N_patches, D) -> (1, H*W, 1024)
            patch_tokens = outputs["x_norm_patchtokens"]
            
            # Reshape back to spatial grid (1, H_grid, W_grid, 1024)
            # input_tensor shape is (1, 3, H_aligned, W_aligned)
            h_aligned, w_aligned = input_tensor.shape[2], input_tensor.shape[3]
            h_grid = h_aligned // self.patch_size
            w_grid = w_aligned // self.patch_size
            
            dense_map = patch_tokens.view(1, h_grid, w_grid, -1)
            
            results.append({
                'feature_map': dense_map, # Keep on GPU for fast indexing later
                'original_size': orig_size, # (H, W)
                'grid_size': (h_grid, w_grid)
            })
            
        return results
