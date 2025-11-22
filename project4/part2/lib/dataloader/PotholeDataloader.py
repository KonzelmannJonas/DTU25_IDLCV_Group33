#!/usr/bin/env python3
"""
PyTorch Dataset for pothole detection using labeled region proposals.

Loads images and their labeled proposal boxes (from VOC-style XML), extracts
region crops, and prepares them for classification training (pothole vs background).
"""
from __future__ import annotations
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class PotholeProposalDataset(Dataset):
    """
    Dataset that loads labeled proposals for pothole detection.
    
    Each sample is a region crop from an image with its corresponding label
    (pothole=1, background=0).
    
    Args:
        images_dir: Directory containing source images
        labels_dir: Directory containing labeled proposal XML files (VOC format)
        image_stems: List of image stems to include (e.g., ['potholes0', 'potholes1'])
        transform: Optional torchvision transform for region crops
        resize: Tuple (height, width) to resize crops, or None for no resize
        balance: If True, balance positive/negative samples by undersampling negatives
    """
    
    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        image_stems: List[str],
        transform: Optional[T.Compose] = None,
        resize: Tuple[int, int] = (64, 64),
        balance: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.resize = resize
        
        # Load all labeled proposals
        self.samples: List[Dict] = []
        for stem in image_stems:
            img_path = self._find_image(stem)
            xml_path = self.labels_dir / f"{stem}_labeled_proposals.xml"
            
            if not img_path.exists() or not xml_path.exists():
                print(f"Warning: Missing image or labels for {stem}, skipping")
                continue
            
            # Parse labeled proposals
            boxes, labels, scores = self._parse_labeled_xml(xml_path)
            
            for bbox, label, score in zip(boxes, labels, scores):
                self.samples.append({
                    'image_path': img_path,
                    'bbox': bbox,
                    'label': label,
                    'score': score,
                    'stem': stem,
                })
        
        # Balance dataset if requested
        if balance:
            self._balance_samples()
        
        print(f"Loaded {len(self.samples)} samples from {len(image_stems)} images")
        pos_count = sum(1 for s in self.samples if s['label'] == 1)
        print(f"  Positives: {pos_count}, Negatives: {len(self.samples) - pos_count}")
    
    def _find_image(self, stem: str) -> Path:
        """Find image file by stem (try common extensions)."""
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            p = self.images_dir / f"{stem}{ext}"
            if p.exists():
                return p
        return self.images_dir / f"{stem}.png"  # default
    
    def _parse_labeled_xml(self, xml_path: Path) -> Tuple[List, List, List]:
        """Parse labeled proposals XML file."""
        root = ET.parse(xml_path).getroot()
        boxes = []
        labels = []
        scores = []
        
        for obj in root.findall('object'):
            name = obj.findtext('name', default='background')
            score_text = obj.findtext('score', default='0.0')
            try:
                score = float(score_text)
            except ValueError:
                score = 0.0
            
            bb = obj.find('bndbox')
            if bb is None:
                continue
            
            xmin = int(float(bb.findtext('xmin', '0')))
            ymin = int(float(bb.findtext('ymin', '0')))
            xmax = int(float(bb.findtext('xmax', '0')))
            ymax = int(float(bb.findtext('ymax', '0')))
            
            # Convert label to binary: pothole=1, background=0
            label = 1 if name.lower() == 'pothole' else 0
            
            boxes.append((xmin, ymin, xmax, ymax))
            labels.append(label)
            scores.append(score)
        
        return boxes, labels, scores
    
    def _balance_samples(self):
        """Balance positive and negative samples by undersampling negatives."""
        positives = [s for s in self.samples if s['label'] == 1]
        negatives = [s for s in self.samples if s['label'] == 0]
        
        if len(positives) == 0:
            return
        
        # Keep all positives, sample negatives to match
        n_pos = len(positives)
        n_neg = min(len(negatives), n_pos * 3)  # Allow 3x more negatives
        
        # Randomly sample negatives
        rng = np.random.default_rng(42)
        neg_indices = rng.choice(len(negatives), size=n_neg, replace=False)
        negatives = [negatives[i] for i in neg_indices]
        
        self.samples = positives + negatives
        rng.shuffle(self.samples)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            crop: Tensor of shape (C, H, W) - RGB image crop
            label: Integer 0 (background) or 1 (pothole)
        """
        sample = self.samples[idx]
        
        # Load image
        img = cv2.imread(str(sample['image_path']))
        if img is None:
            raise RuntimeError(f"Failed to load image: {sample['image_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract crop
        x1, y1, x2, y2 = sample['bbox']
        h, w = img.shape[:2]
        
        # Clamp coordinates
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        
        # Handle invalid boxes
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, min(32, w), min(32, h)
        
        crop = img[y1:y2, x1:x2]
        
        # Resize if specified
        if self.resize is not None:
            crop = cv2.resize(crop, (self.resize[1], self.resize[0]), 
                            interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor (C, H, W) and normalize to [0, 1]
        crop = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        
        # Apply transforms if any
        if self.transform is not None:
            crop = self.transform(crop)
        
        label = sample['label']
        
        return crop, label


def get_dataloaders(
    images_dir: str | Path,
    labels_dir: str | Path,
    train_stems: List[str],
    val_stems: Optional[List[str]] = None,
    batch_size: int = 32,
    resize: Tuple[int, int] = (64, 64),
    balance_train: bool = True,
    num_workers: int = 2,
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """
    Create train (and optionally validation) dataloaders.
    
    Args:
        images_dir: Directory with images
        labels_dir: Directory with labeled proposal XMLs
        train_stems: List of image stems for training
        val_stems: Optional list of stems for validation
        batch_size: Batch size for dataloaders
        resize: Crop resize dimensions
        balance_train: Whether to balance train set
        num_workers: Number of dataloader workers
    
    Returns:
        train_loader, val_loader (or None if no val_stems)
    """
    # Basic transforms (normalize to standard ImageNet stats)
    transform = T.Compose([
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = PotholeProposalDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        image_stems=train_stems,
        transform=transform,
        resize=resize,
        balance=balance_train,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = None
    if val_stems is not None and len(val_stems) > 0:
        val_dataset = PotholeProposalDataset(
            images_dir=images_dir,
            labels_dir=labels_dir,
            image_stems=val_stems,
            transform=transform,
            resize=resize,
            balance=False,  # Don't balance validation
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return train_loader, val_loader
