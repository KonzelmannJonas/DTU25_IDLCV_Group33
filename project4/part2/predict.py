#!/usr/bin/env python3
"""
Prediction script for pothole detection.

Loads a trained classifier and runs inference on test images with region proposals.
Applies Non-Maximum Suppression (NMS) to filter overlapping detections and outputs
predictions as VOC-style XML files with visualization.

Usage:
    python part2/predict.py \
        --images-dir /dtu/datasets1/02516/potholes \
        --proposals-dir proposals \
        --checkpoint outputs/checkpoints/best_model.pth \
        --splits splits.json \
        --conf-threshold 0.5 \
        --nms-threshold 0.3 \
        --output-dir outputs/predictions
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import xml.etree.ElementTree as ET
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.model.SimpleCNN import get_model


def parse_proposals_xml(xml_path: Path) -> Tuple[str, int, int, List[Tuple[int, int, int, int]]]:
    """Parse proposals XML file."""
    root = ET.parse(xml_path).getroot()
    filename = root.findtext('filename', default=xml_path.stem + '.png')
    width = root.find('size/width')
    height = root.find('size/height')
    W = int(width.text) if width is not None and width.text else None
    H = int(height.text) if height is not None and height.text else None
    
    boxes = []
    for obj in root.findall('object'):
        bb = obj.find('bndbox')
        if bb is None:
            continue
        xmin = int(float(bb.findtext('xmin', '0')))
        ymin = int(float(bb.findtext('ymin', '0')))
        xmax = int(float(bb.findtext('xmax', '0')))
        ymax = int(float(bb.findtext('ymax', '0')))
        boxes.append((xmin, ymin, xmax, ymax))
    
    return filename, W, H, boxes


def extract_crop(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    resize: Tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Extract and resize a crop from image."""
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    
    # Clamp coordinates
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    
    # Handle invalid boxes
    if x2 <= x1 or y2 <= y1:
        return np.zeros((resize[0], resize[1], 3), dtype=np.uint8)
    
    crop = img[y1:y2, x1:x2]
    crop = cv2.resize(crop, (resize[1], resize[0]), interpolation=cv2.INTER_LINEAR)
    
    return crop


def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h
    
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    
    return inter / union if union > 0 else 0.0


def non_maximum_suppression(
    boxes: List[Tuple[int, int, int, int]],
    scores: List[float],
    iou_threshold: float = 0.3,
) -> List[int]:
    """
    Apply Non-Maximum Suppression to filter overlapping boxes.
    
    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []
    
    # Sort by score (descending)
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    keep = []
    while len(indices) > 0:
        # Keep highest scoring box
        current = indices[0]
        keep.append(current)
        indices = indices[1:]
        
        # Remove boxes with high IoU
        filtered = []
        for idx in indices:
            if compute_iou(boxes[current], boxes[idx]) < iou_threshold:
                filtered.append(idx)
        indices = filtered
    
    return keep


def write_predictions_xml(
    out_path: Path,
    image_filename: str,
    width: int,
    height: int,
    detections: List[Tuple[Tuple[int, int, int, int], float]],
) -> None:
    """Write predictions to VOC-style XML."""
    ann = ET.Element('annotation')
    ET.SubElement(ann, 'filename').text = image_filename
    size_el = ET.SubElement(ann, 'size')
    ET.SubElement(size_el, 'width').text = str(width)
    ET.SubElement(size_el, 'height').text = str(height)
    ET.SubElement(size_el, 'depth').text = '3'
    ET.SubElement(ann, 'segmented').text = '0'
    
    for bbox, score in detections:
        x1, y1, x2, y2 = bbox
        obj_el = ET.SubElement(ann, 'object')
        ET.SubElement(obj_el, 'name').text = 'pothole'
        ET.SubElement(obj_el, 'pose').text = 'Unspecified'
        ET.SubElement(obj_el, 'truncated').text = '0'
        ET.SubElement(obj_el, 'difficult').text = '0'
        ET.SubElement(obj_el, 'confidence').text = f'{score:.4f}'
        bb_el = ET.SubElement(obj_el, 'bndbox')
        ET.SubElement(bb_el, 'xmin').text = str(int(x1))
        ET.SubElement(bb_el, 'ymin').text = str(int(y1))
        ET.SubElement(bb_el, 'xmax').text = str(int(x2))
        ET.SubElement(bb_el, 'ymax').text = str(int(y2))
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(ann).write(str(out_path), encoding='utf-8', xml_declaration=True)


def visualize_predictions(
    img: np.ndarray,
    detections: List[Tuple[Tuple[int, int, int, int], float]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on image."""
    vis = img.copy()
    
    for bbox, score in detections:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label = f'pothole: {score:.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Background rectangle for text
        ty = max(0, y1 - 4)
        cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 4, ty), color, -1)
        cv2.putText(vis, label, (x1 + 2, ty - 4), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    
    return vis


def predict_image(
    img_path: Path,
    proposals: List[Tuple[int, int, int, int]],
    model: torch.nn.Module,
    device: torch.device,
    resize: Tuple[int, int],
    conf_threshold: float,
    nms_threshold: float,
) -> List[Tuple[Tuple[int, int, int, int], float]]:
    """Run inference on one image."""
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to load image: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Prepare transform
    transform = T.Compose([
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Extract crops and classify
    boxes_to_classify = []
    crops = []
    
    for bbox in proposals:
        crop = extract_crop(img_rgb, bbox, resize)
        crop_tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        crop_tensor = transform(crop_tensor)
        crops.append(crop_tensor)
        boxes_to_classify.append(bbox)
    
    if len(crops) == 0:
        return []
    
    # Batch inference
    crops_batch = torch.stack(crops).to(device)
    
    with torch.no_grad():
        outputs = model(crops_batch)
        probs = F.softmax(outputs, dim=1)
        pothole_probs = probs[:, 1].cpu().numpy()  # Class 1 = pothole
    
    # Filter by confidence
    detections = []
    for bbox, score in zip(boxes_to_classify, pothole_probs):
        if score >= conf_threshold:
            detections.append((bbox, float(score)))
    
    # Apply NMS
    if len(detections) > 0:
        boxes = [d[0] for d in detections]
        scores = [d[1] for d in detections]
        keep_indices = non_maximum_suppression(boxes, scores, nms_threshold)
        detections = [detections[i] for i in keep_indices]
    
    return detections


def main():
    parser = argparse.ArgumentParser(description='Run inference on test images')
    
    # Data arguments
    parser.add_argument('--images-dir', required=True, help='Directory with images')
    parser.add_argument('--proposals-dir', required=True, help='Directory with proposal XMLs')
    parser.add_argument('--splits', default='splits.json', help='JSON file with train/test splits')
    
    # Model arguments
    parser.add_argument('--checkpoint', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--model', choices=['simple', 'resnet18'], default='simple', help='Model architecture')
    parser.add_argument('--resize', type=int, default=64, help='Resize crops to this size')
    
    # Inference arguments
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='Confidence threshold for detections')
    parser.add_argument('--nms-threshold', type=float, default=0.3, help='NMS IoU threshold')
    
    # Output arguments
    parser.add_argument('--output-dir', required=True, help='Directory to save predictions')
    parser.add_argument('--visualize', action='store_true', help='Save visualization images')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.visualize:
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load splits
    print(f"\nLoading splits from {args.splits}...")
    with open(args.splits, 'r') as f:
        splits = json.load(f)
    
    test_stems = splits['test']
    print(f"Processing {len(test_stems)} test images")
    
    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create model (try to get config from checkpoint)
    model_type = checkpoint.get('args', {}).get('model', args.model)
    model = get_model(model_type, num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")
    if 'val_acc' in checkpoint:
        print(f"Validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Process each test image
    images_dir = Path(args.images_dir)
    proposals_dir = Path(args.proposals_dir)
    resize = (args.resize, args.resize)
    
    print(f"\nProcessing {len(test_stems)} test images...")
    
    all_results = {}
    
    for stem in test_stems:
        # Find image
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            p = images_dir / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break
        
        if img_path is None:
            print(f"Warning: Image not found for {stem}, skipping")
            continue
        
        # Load proposals
        proposal_xml = proposals_dir / f"{stem}_proposals.xml"
        if not proposal_xml.exists():
            print(f"Warning: Proposals not found for {stem}, skipping")
            continue
        
        filename, W, H, proposals = parse_proposals_xml(proposal_xml)
        
        print(f"\n{stem}: {len(proposals)} proposals")
        
        # Run inference
        detections = predict_image(
            img_path,
            proposals,
            model,
            device,
            resize,
            args.conf_threshold,
            args.nms_threshold,
        )
        
        print(f"  → {len(detections)} detections after NMS")
        
        # Save predictions XML
        pred_xml = output_dir / f"{stem}_predictions.xml"
        write_predictions_xml(pred_xml, filename, W or 0, H or 0, detections)
        print(f"  Saved: {pred_xml.name}")
        
        # Visualize if requested
        if args.visualize:
            img = cv2.imread(str(img_path))
            vis = visualize_predictions(img, detections)
            vis_path = vis_dir / f"{stem}_detections.png"
            cv2.imwrite(str(vis_path), vis)
            print(f"  Visualization: {vis_path.name}")
        
        # Store results
        all_results[stem] = {
            'num_proposals': len(proposals),
            'num_detections': len(detections),
            'detections': [
                {'bbox': list(bbox), 'confidence': float(score)}
                for bbox, score in detections
            ]
        }
    
    # Save summary
    import json
    summary_path = output_dir / 'predictions_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved summary: {summary_path}")
    
    print(f"\n{'='*60}")
    print("Inference complete!")
    print('='*60)


if __name__ == '__main__':
    main()
