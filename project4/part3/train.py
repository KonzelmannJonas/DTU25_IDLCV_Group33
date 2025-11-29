#!/usr/bin/env python3
"""
Training script for pothole detection classifier.

Trains a CNN to classify region proposals as pothole (class 1) or background (class 0).

Usage:
    python part2/train.py \
        --images-dir part1/data/examples/images \
        --labels-dir part1/proposals_max200/labeled_proposals \
        --train-stems potholes0 potholes1 potholes2 \
        --val-stems potholes3 \
        --model simple \
        --epochs 20 \
        --batch-size 32 \
        --lr 0.001 \
        --output-dir part2/checkpoints
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.dataloader.PotholeDataloader import get_dataloaders
from lib.model.SimpleCNN import get_model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    
    return {'loss': avg_loss, 'accuracy': accuracy}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # For computing precision/recall
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Confusion matrix elements (class 1 = pothole is positive)
            for pred, label in zip(predicted, labels):
                if label == 1 and pred == 1:
                    true_positives += 1
                elif label == 0 and pred == 1:
                    false_positives += 1
                elif label == 1 and pred == 0:
                    false_negatives += 1
                else:  # label == 0 and pred == 0
                    true_negatives += 1
    
    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    
    # Precision and recall
    precision = 100.0 * true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = 100.0 * true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': true_positives,
        'fp': false_positives,
        'tn': true_negatives,
        'fn': false_negatives,
    }


def main():
    parser = argparse.ArgumentParser(description='Train pothole detection classifier')
    
    # Data arguments
    parser.add_argument('--images-dir', required=True, help='Directory with images')
    parser.add_argument('--labels-dir', required=True, help='Directory with labeled proposal XMLs')
    parser.add_argument('--splits', default='splits.json', help='JSON file with train/test splits')
    parser.add_argument('--val-split', type=float, default=0.1, help='Fraction of train data to use for validation')
    
    # Model arguments
    parser.add_argument('--model', choices=['simple', 'resnet18'], default='simple', help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights (for resnet18)')
    parser.add_argument('--freeze-backbone', action='store_true', help='Freeze backbone (for resnet18)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (for simple CNN)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--resize', type=int, default=64, help='Resize crops to this size (square)')
    parser.add_argument('--balance-train', action='store_true', default=True, help='Balance training data')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of dataloader workers')
    
    # Output arguments
    parser.add_argument('--output-dir', required=True, help='Directory to save checkpoints and logs')
    parser.add_argument('--save-every', type=int, default=5, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load splits
    print(f"\nLoading splits from {args.splits}...")
    with open(args.splits, 'r') as f:
        splits = json.load(f)
    
    train_stems = splits['train']
    
    # Split train into train and val
    if args.val_split > 0:
        import random
        random.seed(42)
        train_stems_shuffled = train_stems.copy()
        random.shuffle(train_stems_shuffled)
        val_size = int(len(train_stems_shuffled) * args.val_split)
        val_stems = train_stems_shuffled[:val_size]
        train_stems = train_stems_shuffled[val_size:]
        print(f"Split train data: {len(train_stems)} train, {len(val_stems)} val")
    else:
        val_stems = None
        print(f"Using {len(train_stems)} images for training (no validation)")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = get_dataloaders(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        train_stems=train_stems,
        val_stems=val_stems,
        batch_size=args.batch_size,
        resize=(args.resize, args.resize),
        balance_train=args.balance_train,
        num_workers=args.num_workers,
    )
    
    # Create model
    print(f"\nCreating model: {args.model}")
    model_kwargs = {}
    if args.model == 'simple':
        model_kwargs['dropout'] = args.dropout
    elif args.model == 'resnet18':
        model_kwargs['pretrained'] = args.pretrained
        model_kwargs['freeze_backbone'] = args.freeze_backbone
    
    model = get_model(args.model, num_classes=2, **model_kwargs)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_val_acc = 0.0
    history = {'train': [], 'val': []}
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print('='*60)
        
        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        history['train'].append(train_metrics)
        
        # Validate
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, device)
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            print(f"      - Precision: {val_metrics['precision']:.2f}%, Recall: {val_metrics['recall']:.2f}%, F1: {val_metrics['f1']:.2f}%")
            history['val'].append(val_metrics)
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_path = output_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': best_val_acc,
                    'args': vars(args),
                }, best_path)
                print(f"✓ Saved best model (val_acc={best_val_acc:.2f}%)")
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
            }, checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path.name}")
    
    # Save final model
    final_path = output_dir / 'final_model.pth'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
    }, final_path)
    print(f"\n✓ Saved final model: {final_path}")
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Saved training history: {history_path}")
    
    print(f"\n{'='*60}")
    print("Training complete!")
    if val_loader is not None:
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print('='*60)


if __name__ == '__main__':
    main()
