# Project 4 Part 2: Pothole Detection - Instructions

## Overview
This project trains a CNN classifier to detect potholes in images using region proposals.

## Dataset Information
- **Images**: `/dtu/datasets1/02516/potholes/` (potholes0.png to potholes642.png)
- **Annotations**: `/dtu/datasets1/02516/potholes/annotations/` (XML files with ground truth)
- **Total Images**: 643 images
- **Split**: 80% train (~514 images), 20% test (~129 images)

## Setup

### 1. Install Dependencies
```bash
# Install OpenCV for selective search
pip install opencv-python opencv-contrib-python

# Install PyTorch and torchvision (if not already installed)
pip install torch torchvision

# Install other dependencies
pip install tqdm Pillow
```

### 2. Generate Data (Run Once)
This will create splits, generate proposals, and label them:

```bash
cd project4/part2
make prepare-data
```

This executes three steps:
1. **Generate splits** (`splits.json`): 80/20 train/test split
2. **Generate proposals**: Uses Selective Search to create ~200 region proposals per image
3. **Label proposals**: Labels each proposal as "pothole" or "background" based on IoU with ground truth

**Expected output:**
- `splits.json`: Train/test split information
- `proposals/`: Directory with proposal XML files
- `labeled_proposals/`: Directory with labeled proposal XML files

## Training

### Quick Start
```bash
make train
```

### Custom Training
```bash
python train.py \
    --images-dir /dtu/datasets1/02516/potholes \
    --labels-dir labeled_proposals \
    --splits splits.json \
    --model simple \
    --epochs 20 \
    --batch-size 32 \
    --lr 0.001 \
    --resize 64 \
    --output-dir outputs/checkpoints \
    --balance-train
```

### Training with ResNet18 (Pretrained)
```bash
make train-resnet

# Or manually:
python train.py \
    --images-dir /dtu/datasets1/02516/potholes \
    --labels-dir labeled_proposals \
    --splits splits.json \
    --model resnet18 \
    --pretrained \
    --epochs 15 \
    --batch-size 32 \
    --lr 0.0001 \
    --resize 64 \
    --output-dir outputs/checkpoints_resnet
```

### Training Output
- `outputs/checkpoints/best_model.pth`: Best model based on validation accuracy
- `outputs/checkpoints/final_model.pth`: Final model after all epochs
- `outputs/checkpoints/training_history.json`: Training metrics

## Prediction/Inference

### Run Predictions
```bash
make predict
```

### With Visualizations
```bash
make visualize
```

### Custom Prediction
```bash
python predict.py \
    --images-dir /dtu/datasets1/02516/potholes \
    --proposals-dir proposals \
    --checkpoint outputs/checkpoints/best_model.pth \
    --splits splits.json \
    --conf-threshold 0.5 \
    --nms-threshold 0.3 \
    --output-dir outputs/predictions \
    --visualize
```

### Prediction Output
- `outputs/predictions/*.xml`: Detection results in VOC XML format
- `outputs/predictions/visualizations/*.png`: Visualizations (if --visualize flag is used)
- `outputs/predictions/summary.json`: Summary statistics

## Model Architectures

### SimpleCNN (Default)
- 3 convolutional blocks
- Dropout for regularization
- Fast training, good for quick experiments

### ResNet18
- Pretrained on ImageNet
- Transfer learning
- Better accuracy, slower training

## Parameters

### Data Generation
- `--max-proposals`: Number of proposals per image (default: 200)
- `--iou-threshold`: IoU threshold for labeling (default: 0.5)

### Training
- `--model`: Model architecture (`simple` or `resnet18`)
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--resize`: Crop resize size (default: 64x64)
- `--balance-train`: Balance positive/negative samples
- `--val-split`: Validation split from train data (default: 0.1)

### Prediction
- `--conf-threshold`: Confidence threshold for detections (default: 0.5)
- `--nms-threshold`: NMS IoU threshold (default: 0.3)
- `--visualize`: Save visualization images

## Complete Workflow

```bash
# 1. Generate data (run once)
make prepare-data

# 2. Train model
make train

# 3. Run predictions with visualizations
make visualize

# 4. Check results
ls outputs/predictions/visualizations/
```

## Troubleshooting

### Issue: "opencv-contrib-python not installed"
```bash
pip install opencv-contrib-python
```

### Issue: "Out of memory"
- Reduce `--batch-size` (try 16 or 8)
- Reduce `--resize` (try 32)

### Issue: "No proposals generated"
- Check if images exist in `/dtu/datasets1/02516/potholes/`
- Verify Selective Search is working: `python -c "import cv2; print(cv2.ximgproc)"`

### Issue: "Poor detection results"
- Increase training epochs
- Use pretrained ResNet18: `make train-resnet`
- Adjust thresholds: lower `--conf-threshold`, adjust `--nms-threshold`

## Clean Up

```bash
# Remove outputs only
make clean

# Remove everything (including generated data)
make clean-all
```

## File Structure
```
part2/
├── generate_splits.py          # Generate train/test splits
├── generate_proposals.py       # Generate region proposals
├── label_proposals.py          # Label proposals with ground truth
├── train.py                    # Training script
├── predict.py                  # Prediction script
├── Makefile                    # Build automation
├── INSTRUCTIONS.md            # This file
├── splits.json                 # Train/test splits (generated)
├── proposals/                  # Proposal XMLs (generated)
├── labeled_proposals/          # Labeled XMLs (generated)
├── outputs/                    # Training outputs
└── lib/
    ├── dataloader/
    │   └── PotholeDataloader.py
    └── model/
        └── SimpleCNN.py
```
