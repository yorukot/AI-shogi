import os
import ultralytics
from ultralytics import YOLO
import torch
print(os.getcwd())
print(ultralytics.__version__)

# Set PyTorch memory allocation to be more efficient
torch.cuda.empty_cache()
# Optional: Set memory optimization configuration
# torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of available GPU memory

root_dir = './'
output_path = './output'
yaml_dir = './data.yaml'

train_path = os.path.join(root_dir, 'train', 'images')
valid_path = os.path.join(root_dir, 'valid', 'images')

if __name__ == '__main__':
    model = YOLO('yolo11s.pt')
    results = model.train(
        device=0,
        data=yaml_dir,
        epochs=200,
        batch=16,  # Reduced from 32 to balance memory and performance
        lr0=0.0001,
        lrf=0.1,
        imgsz=640,
        plots=True,
        cache=True,  # Add caching to improve memory efficiency
        half=True    # Use half precision (FP16) to save memory
    )
