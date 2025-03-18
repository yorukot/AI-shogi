# YOLO Object Detection Project

## Overview
This project implements object detection using a YOLO (You Only Look Once) model. It includes scripts for training a custom model on your dataset, validating results, and running inference.

## Project Structure

```
    ├── train.py            # Main training script
    ├── main.py             # Entry point/inference script
    ├── data.yaml           # Dataset configuration
    ├── changelabel.py      # Script to modify/transform labels
    ├── train/              # Training dataset
    │   └── images/         # Training images
    ├── valid/              # Validation dataset
    │   └── images/         # Validation images
    │       └── classes.txt # Class definitions
    └── output/             # Training results and model outputs
```

## Requirements
- Python 3.8+
- PyTorch
- Ultralytics YOLO
- CUDA-compatible GPU (for faster training)

## Installation

    # Clone this repository
    git clone <repository-url>
    cd <repository-directory>

    # Install dependencies
    pip install ultralytics torch torchvision

## Dataset Preparation
1. Organize your dataset into train and validation sets
2. Update `data.yaml` with your class names and dataset paths
3. Use `changelabel.py` if you need to transform existing labels

## Training
Run the training script to train your model:

    python train.py

This will train a YOLO model with the following configuration:
- Model: yolo11n.pt
- 200 epochs
- Batch size: 16 (optimized for memory efficiency)
- Image size: 640
- Learning rate: 0.0001
- FP16 precision

Training progress and results will be saved to the `output` directory.

## Memory Optimization
The training script includes memory optimization techniques:
- CUDA memory cache clearing
- Half-precision training (FP16)
- Caching optimization

If you're experiencing memory issues, you can:
- Further reduce batch size
- Uncomment and adjust the memory fraction setting
- Reduce image size
- Use gradient accumulation

## Running Inference
Use `main.py` to run inference on new images:

    python main.py --source <path-to-image-or-directory>

## Customization
- Modify `data.yaml` to change dataset paths or class names
- Adjust training parameters in `train.py` based on your requirements

## License
MIT LICENSE