# Makefile for Project 4 Part 2: Pothole Detection Training & Inference

# Configuration
IMAGES_DIR := /dtu/datasets1/02516/potholes/images
ANNOTATIONS_DIR := /dtu/datasets1/02516/potholes/annotations
PROPOSALS_DIR := proposals
LABELS_DIR := labeled_proposals
OUTPUT_DIR := outputs
SPLITS_FILE := splits.json

# Training configuration
TRAIN_STEMS := potholes0 potholes1 potholes2
VAL_STEMS := potholes3
TEST_STEMS := potholes3

MODEL := simple
EPOCHS := 20
BATCH_SIZE := 32
LR := 0.001
RESIZE := 64

# Prediction configuration
CONF_THRESHOLD := 0.5
NMS_THRESHOLD := 0.3

.PHONY: all train predict visualize clean test-model help prepare-data generate-splits generate-proposals label-proposals

all: prepare-data train predict

prepare-data: generate-splits generate-proposals label-proposals

generate-splits:
	@echo "Generating train/test splits..."
	python generate_splits.py

generate-proposals:
	@echo "Generating region proposals..."
	python generate_proposals.py

label-proposals:
	@echo "Labeling proposals with ground truth..."
	python label_proposals.py

help:
	@echo "Project 4 Part 2: Pothole Detection"
	@echo ""
	@echo "Available targets:"
	@echo "  prepare-data   - Generate splits, proposals, and labels (run this first!)"
	@echo "  generate-splits - Generate train/test splits (80/20)"
	@echo "  generate-proposals - Generate region proposals for all images"
	@echo "  label-proposals - Label proposals with ground truth"
	@echo "  train          - Train the classifier"
	@echo "  predict        - Run inference on test images"
	@echo "  visualize      - Run inference with visualizations"
	@echo "  test-model     - Test model architecture"
	@echo "  clean          - Remove output files"
	@echo "  all            - Prepare data, train and predict (default)"
	@echo ""
	@echo "Configuration:"
	@echo "  MODEL=$(MODEL) (simple or resnet18)"
	@echo "  EPOCHS=$(EPOCHS)"
	@echo "  BATCH_SIZE=$(BATCH_SIZE)"
	@echo "  LR=$(LR)"

train:
	@echo "Training classifier..."
	python train.py \
		--images-dir $(IMAGES_DIR) \
		--labels-dir $(LABELS_DIR) \
		--splits $(SPLITS_FILE) \
		--model $(MODEL) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--lr $(LR) \
		--resize $(RESIZE) \
		--output-dir $(OUTPUT_DIR)/checkpoints \
		--balance-train

predict:
	@echo "Running inference..."
	python predict.py \
		--images-dir $(IMAGES_DIR) \
		--proposals-dir $(PROPOSALS_DIR) \
		--checkpoint $(OUTPUT_DIR)/checkpoints/best_model.pth \
		--splits $(SPLITS_FILE) \
		--model $(MODEL) \
		--resize $(RESIZE) \
		--conf-threshold $(CONF_THRESHOLD) \
		--nms-threshold $(NMS_THRESHOLD) \
		--output-dir $(OUTPUT_DIR)/predictions

visualize:
	@echo "Running inference with visualization..."
	python predict.py \
		--images-dir $(IMAGES_DIR) \
		--proposals-dir $(PROPOSALS_DIR) \
		--checkpoint $(OUTPUT_DIR)/checkpoints/best_model.pth \
		--splits $(SPLITS_FILE) \
		--model $(MODEL) \
		--resize $(RESIZE) \
		--conf-threshold $(CONF_THRESHOLD) \
		--nms-threshold $(NMS_THRESHOLD) \
		--output-dir $(OUTPUT_DIR)/predictions \
		--visualize

test-model:
	@echo "Testing model architecture..."
	python lib/model/SimpleCNN.py

clean:
	@echo "Cleaning output files..."
	rm -rf $(OUTPUT_DIR)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Done."

clean-all: clean
	@echo "Cleaning generated data..."
	rm -rf $(PROPOSALS_DIR) $(LABELS_DIR) $(SPLITS_FILE)
	@echo "Done."

# Quick training with ResNet18
train-resnet:
	@echo "Training with ResNet18 (pretrained)..."
	python train.py \
		--images-dir $(IMAGES_DIR) \
		--labels-dir $(LABELS_DIR) \
		--splits $(SPLITS_FILE) \
		--model resnet18 \
		--pretrained \
		--epochs 15 \
		--batch-size $(BATCH_SIZE) \
		--lr 0.0001 \
		--resize $(RESIZE) \
		--output-dir $(OUTPUT_DIR)/checkpoints_resnet \
		--balance-train
