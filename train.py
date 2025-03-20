import os
import ultralytics
from ultralytics import YOLO
import torch
import gc
print(os.getcwd())
print(ultralytics.__version__)

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA is not available. Training will use CPU, which will be much slower.")

# The most aggressive memory management possible
torch.cuda.empty_cache()
gc.collect()  # Force garbage collection
torch.cuda.set_per_process_memory_fraction(0.8)  # Increased to 80% to allow for larger model
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
    # Use a larger model for better accuracy (if memory allows)
    model = YOLO('yolov8m.pt')  # Using medium-sized model for better accuracy
    
    # Set device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Clear cache before training
    torch.cuda.empty_cache()
    gc.collect()
    results = model.train(
        device=device,
        data=yaml_dir,
        epochs=300,  # Increased epochs for better convergence
        batch=16,  # Smaller batch size to accommodate larger model
        lr0=0.001,  # Higher initial learning rate
        lrf=0.01,  # Lower final LR factor for better fine-tuning
        imgsz=640,  # Increased image size for better detection of small objects
        plots=True,  # Enable plots to monitor training
        cache=True,
        half=True,  # Keep half precision for efficiency
        workers=2,  # Added some workers for faster data loading
        close_mosaic=10,
        nbs=64,  # Nominal batch size for optimizer
        patience=50,  # Early stopping patience
        cos_lr=True,  # Use cosine LR scheduler
        weight_decay=0.0005,  # Add weight decay for regularization
        warmup_epochs=3,  # Add warmup epochs
        mosaic=1.0,  # Enable mosaic augmentation
        mixup=0.1,  # Add mixup augmentation
        degrees=0.1,  # Rotation augmentation
        translate=0.1,  # Translation augmentation
        scale=0.5,  # Scaling augmentation
        fliplr=0.5,  # Horizontal flip augmentation
        flipud=0.0,  # Vertical flip augmentation (usually not helpful for object detection)
        hsv_h=0.015,  # HSV hue augmentation
        hsv_s=0.7,  # HSV saturation augmentation
        hsv_v=0.4,  # HSV value augmentation
        copy_paste=0.1,  # Copy-paste augmentation
        rect=False,  # Rectangular training with aspect ratios
        save=True,  # Save checkpoints
        save_period=10,  # Save checkpoints every 10 epochs
        optimizer='AdamW',  # Use AdamW optimizer
    )

    # Save best model
    final_model_path = os.path.join(output_path, 'best_model.pt')
    if os.path.exists(os.path.join('runs/detect/train', 'weights/best.pt')):
        import shutil
        shutil.copy(os.path.join('runs/detect/train', 'weights/best.pt'), final_model_path)
        print(f"Best model saved to {final_model_path}")
