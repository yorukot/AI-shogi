import os
import ultralytics
from ultralytics import YOLO
import torch
import gc
print(os.getcwd())
print(ultralytics.__version__)

# The most aggressive memory management possible
torch.cuda.empty_cache()
gc.collect()  # Force garbage collection
torch.cuda.set_per_process_memory_fraction(0.6)  # Even more conservative - use only 60% of available GPU memory
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.5'

# Disable gradient caching
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

root_dir = './'
output_path = './output'
yaml_dir = './data.yaml'

train_path = os.path.join(root_dir, 'train', 'images')
valid_path = os.path.join(root_dir, 'valid', 'images')

if __name__ == '__main__':
    # Use smallest available model
    model = YOLO('yolo11s.pt')  # Switch to the smallest YOLOv8 nano model
    
    # Clear cache before training
    torch.cuda.empty_cache()
    gc.collect()
    results = model.train(
        device=0,
        data=yaml_dir,
        epochs=200,
        batch=4,  # Extreme reduction in batch size
        lr0=0.0001,
        lrf=0.1,
        imgsz=416,  # Further reduced image size
        plots=False,  # Disable plotting to save memory
        cache=True,
        half=True,
        workers=0,
        close_mosaic=10,
        nbs=64,  # Nominal batch size for optimizer
        patience=50,  # Early stopping patience
    )
