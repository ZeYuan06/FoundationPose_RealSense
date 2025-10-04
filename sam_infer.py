import torch
import numpy as np
from PIL import Image
import cv2
import argparse
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os

class SAMInference:
    def __init__(self, ckpt_path, model_cfg):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, ckpt_path, device=self.device))
        self.image = None
        self.original_image = None
        self.points = []
        self.labels = []
        self.current_mask = None  # Store the current predicted mask
        
    def load_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Load once and convert properly
        pil_image = Image.open(image_path).convert("RGB")
        self.image = np.array(pil_image)
        self.original_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            self.predictor.set_image(self.image)
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y])
            self.labels.append(1)  # Positive point
            print(f"Added point at ({x}, {y})")
            self.update_visualization()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points.append([x, y])
            self.labels.append(0)  # Negative point
            print(f"Added negative point at ({x}, {y})")
            self.update_visualization()
    
    def update_visualization(self):
        if not self.points:
            self.current_mask = None
            return
            
        img_copy = self.original_image.copy()
        
        # Draw points
        for (point, label) in zip(self.points, self.labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(img_copy, tuple(point), 5, color, -1)
        
        # Generate and overlay mask
        try:
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                masks, scores, _ = self.predictor.predict(
                    point_coords=np.array(self.points),
                    point_labels=np.array(self.labels),
                    multimask_output=True
                )
            
            # Use the best mask and store it
            best_mask = masks[np.argmax(scores)].astype(bool)
            self.current_mask = best_mask  # Store the current mask
            
            # Create colored overlay
            mask_overlay = np.zeros_like(img_copy)
            mask_overlay[best_mask] = [0, 255, 255]  # Yellow mask
            img_copy = cv2.addWeighted(img_copy, 0.7, mask_overlay, 0.3, 0)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            self.current_mask = None
        
        cv2.imshow("SAM2 Inference", img_copy)

    def save_mask(self, save_path="mask.png"):
        """Save the current predicted mask to file"""
        if self.current_mask is None:
            print("No mask available to save. Please make some predictions first.")
            return False
        
        try:
            # Convert to 0-255 for saving
            mask_image = (self.current_mask * 255).astype(np.uint8)
            
            # Save the mask
            cv2.imwrite(save_path, mask_image)
            print(f"Mask saved to: {save_path}")
            return True
            
        except Exception as e:
            print(f"Error saving mask: {e}")
            return False
    
    def run_interactive(self, image_path):
        try:
            self.load_image(image_path)
            
            cv2.imshow("SAM2 Inference", self.original_image)
            cv2.setMouseCallback("SAM2 Inference", self.mouse_callback)
            
            print("Left click: Add positive point")
            print("Right click: Add negative point") 
            print("Press 'r' to reset, 's' to save mask, 'q' to quit")
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.points = []
                    self.labels = []
                    self.current_mask = None
                    cv2.imshow("SAM2 Inference", self.original_image)
                    print("Reset points")
                elif key == ord('s'):
                    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.save_mask(f"mask_{timestamp}.png")
                    
        finally:
            cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_path', type=str, default='mask.png', help='Path to save mask')
    parser.add_argument('--ckpt_path', type=str, default='./third_party/sam2/checkpoints/sam2.1_hiera_large.pt')
    parser.add_argument('--model_cfg', type=str, default='configs/sam2.1/sam2.1_hiera_l.yaml')
    args = parser.parse_args()

    sam_inference = SAMInference(args.ckpt_path, args.model_cfg)
    sam_inference.run_interactive(args.image_path)

    if sam_inference.current_mask is not None:
        sam_inference.save_mask(args.output_path)
