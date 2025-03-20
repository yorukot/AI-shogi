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
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("WARNING: CUDA is not available. Training will use CPU, which will be much slower.")

# The most aggressive memory management possible
torch.cuda.empty_cache()
gc.collect()  # Force garbage collection
torch.cuda.set_per_process_memory_fraction(0.9)  # Use up to 90% of available memory
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.6'

# Disable gradient caching
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Add memory-saving configuration
torch.set_float32_matmul_precision('medium')  # Reduce precision for matrix multiplications

root_dir = './'
output_path = './output'
yaml_dir = './data.yaml'

train_path = os.path.join(root_dir, 'train', 'images')
valid_path = os.path.join(root_dir, 'valid', 'images')

if __name__ == '__main__':
    # Use the smallest model to save memory
    model = YOLO('yolov8n.pt')  # Using nano model for memory efficiency
    
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
        batch=8,  # Drastically reduced batch size to save memory
        lr0=0.001,  # Higher initial learning rate
        lrf=0.01,  # Lower final LR factor for better fine-tuning
        imgsz=416,  # Reduced image size to save memory
        plots=False,  # Disable plots to save memory
        cache=False,  # Disable cache to reduce memory usage
        half=True,  # Keep half precision for efficiency
        workers=0,  # No additional workers to save memory
        close_mosaic=10,
        nbs=64,  # Nominal batch size for optimizer
        patience=50,  # Early stopping patience
        cos_lr=True,  # Use cosine LR scheduler
        weight_decay=0.0005,  # Add weight decay for regularization
        warmup_epochs=3,  # Add warmup epochs
        mosaic=0.5,  # Reduce mosaic augmentation
        mixup=0.0,  # Disable mixup augmentation to save memory
        degrees=0.1,  # Rotation augmentation
        translate=0.1,  # Translation augmentation
        scale=0.5,  # Scaling augmentation
        fliplr=0.5,  # Horizontal flip augmentation
        flipud=0.0,  # Vertical flip augmentation (usually not helpful for object detection)
        hsv_h=0.015,  # HSV hue augmentation
        hsv_s=0.7,  # HSV saturation augmentation
        hsv_v=0.4,  # HSV value augmentation
        copy_paste=0.0,  # Disable copy-paste augmentation to save memory
        rect=True,  # Use rectangular training to optimize memory usage
        save=True,  # Save checkpoints
        save_period=20,  # Save checkpoints less frequently
        optimizer='SGD',  # Use memory-efficient SGD optimizer instead of AdamW
    )

    # Save best model
    final_model_path = os.path.join(output_path, 'best_model.pt')
    if os.path.exists(os.path.join('runs/detect/train', 'weights/best.pt')):
        import shutil
        shutil.copy(os.path.join('runs/detect/train', 'weights/best.pt'), final_model_path)
        print(f"Best model saved to {final_model_path}")
