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


def parse_ground_truth_xml(xml_path: Path) -> List[Tuple[int, int, int, int]]:
    """Parse ground truth annotation XML file."""
    if not xml_path.exists():
        return []
    
    root = ET.parse(xml_path).getroot()
    boxes = []
    
    for obj in root.findall('object'):
        name = obj.findtext('name', '')
        if name.lower() != 'pothole':
            continue
        
        bb = obj.find('bndbox')
        if bb is None:
            continue
        
        xmin = int(float(bb.findtext('xmin', '0')))
        ymin = int(float(bb.findtext('ymin', '0')))
        xmax = int(float(bb.findtext('xmax', '0')))
        ymax = int(float(bb.findtext('ymax', '0')))
        boxes.append((xmin, ymin, xmax, ymax))
    
    return boxes


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


def visualize_comparison(
    img: np.ndarray,
    predictions: List[Tuple[Tuple[int, int, int, int], float]],
    ground_truth: List[Tuple[int, int, int, int]],
    pred_color: Tuple[int, int, int] = (0, 255, 0),  # Green
    gt_color: Tuple[int, int, int] = (255, 0, 0),     # Blue (BGR format)
    thickness: int = 2,
) -> np.ndarray:
    """Draw both predictions and ground truth boxes on image."""
    vis = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    # Draw ground truth boxes (blue)
    for bbox in ground_truth:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), gt_color, thickness)
        
        # Draw label
        label = 'GT'
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        ty = max(0, y1 - 4)
        cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 4, ty), gt_color, -1)
        cv2.putText(vis, label, (x1 + 2, ty - 4), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    # Draw predictions (green)
    for bbox, score in predictions:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), pred_color, thickness)
        
        # Draw label
        label = f'Pred: {score:.2f}'
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        ty = max(0, y2 + th + 4)
        cv2.rectangle(vis, (x1, y2), (x1 + tw + 4, ty + 4), pred_color, -1)
        cv2.putText(vis, label, (x1 + 2, y2 + th + 2), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    
    return vis


def compute_detection_metrics(
    predictions: List[Tuple[Tuple[int, int, int, int], float]],
    ground_truth: List[Tuple[int, int, int, int]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute detection metrics (precision, recall, F1) based on IoU matching.
    
    Args:
        predictions: List of (bbox, confidence) tuples
        ground_truth: List of ground truth bboxes
        iou_threshold: IoU threshold for considering a match
    
    Returns:
        Dictionary with TP, FP, FN, precision, recall, F1, and average IoU
    """
    pred_boxes = [p[0] for p in predictions]
    
    # Track which GT boxes have been matched
    gt_matched = [False] * len(ground_truth)
    pred_matched = [False] * len(pred_boxes)
    
    true_positives = 0
    total_iou = 0.0
    
    # For each prediction, find best matching GT box
    for pred_idx, pred_box in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(ground_truth):
            if gt_matched[gt_idx]:
                continue
            
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # If best IoU exceeds threshold, it's a match
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            true_positives += 1
            total_iou += best_iou
            gt_matched[best_gt_idx] = True
            pred_matched[pred_idx] = True
    
    false_positives = len(pred_boxes) - true_positives
    false_negatives = len(ground_truth) - true_positives
    
    precision = true_positives / max(1, len(pred_boxes))
    recall = true_positives / max(1, len(ground_truth)) if len(ground_truth) > 0 else 0.0
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    avg_iou = total_iou / max(1, true_positives)
    
    return {
        'TP': true_positives,
        'FP': false_positives,
        'FN': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_iou': avg_iou,
        'num_predictions': len(pred_boxes),
        'num_ground_truth': len(ground_truth),
    }


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
    parser.add_argument('--annotations-dir', default=None, help='Directory with ground truth annotations')
    parser.add_argument('--splits', default='splits.json', help='JSON file with train/test splits')
    
    # Model arguments
    parser.add_argument('--checkpoint', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--model', choices=['simple', 'resnet18'], default='simple', help='Model architecture')
    parser.add_argument('--resize', type=int, default=64, help='Resize crops to this size')
    
    # Inference arguments
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='Confidence threshold for detections')
    parser.add_argument('--nms-threshold', type=float, default=0.3, help='NMS IoU threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='IoU threshold for detection matching')
    
    # Output arguments
    parser.add_argument('--output-dir', required=True, help='Directory to save predictions')
    parser.add_argument('--visualize', action='store_true', help='Save visualization images')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_dir = output_dir / 'comparisons'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if annotations directory is provided
    annotations_dir = None
    if args.annotations_dir:
        annotations_dir = Path(args.annotations_dir)
        if not annotations_dir.exists():
            print(f"Warning: Annotations directory not found: {annotations_dir}")
            annotations_dir = None
        else:
            print(f"Ground truth annotations: {annotations_dir}")
    
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
    overall_metrics = {
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'total_iou': 0.0,
        'num_matched': 0,
    }
    
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
        
        # Load ground truth if available
        ground_truth = []
        metrics = None
        if annotations_dir:
            gt_xml = annotations_dir / f"{stem}.xml"
            if gt_xml.exists():
                ground_truth = parse_ground_truth_xml(gt_xml)
                print(f"  Ground truth: {len(ground_truth)} potholes")
                
                # Compute metrics
                metrics = compute_detection_metrics(detections, ground_truth, args.iou_threshold)
                print(f"  Metrics: P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                      f"F1={metrics['f1']:.3f} IoU={metrics['avg_iou']:.3f}")
                
                # Update overall metrics
                overall_metrics['TP'] += metrics['TP']
                overall_metrics['FP'] += metrics['FP']
                overall_metrics['FN'] += metrics['FN']
                if metrics['TP'] > 0:
                    overall_metrics['total_iou'] += metrics['avg_iou'] * metrics['TP']
                    overall_metrics['num_matched'] += metrics['TP']
        
        # Save predictions XML
        pred_xml = output_dir / f"{stem}_predictions.xml"
        write_predictions_xml(pred_xml, filename, W or 0, H or 0, detections)
        print(f"  Saved: {pred_xml.name}")
        
        # Always visualize predictions
        img = cv2.imread(str(img_path))
        vis = visualize_predictions(img, detections)
        vis_path = vis_dir / f"{stem}_detections.png"
        cv2.imwrite(str(vis_path), vis)
        
        # If ground truth available, create comparison visualization
        if len(ground_truth) > 0:
            comp_vis = visualize_comparison(img, detections, ground_truth)
            comp_path = comparison_dir / f"{stem}_comparison.png"
            cv2.imwrite(str(comp_path), comp_vis)
            print(f"  Comparison: {comp_path.name}")
        
        # Store results
        result_entry = {
            'num_proposals': len(proposals),
            'num_detections': len(detections),
            'detections': [
                {'bbox': list(bbox), 'confidence': float(score)}
                for bbox, score in detections
            ]
        }
        
        if metrics:
            result_entry['metrics'] = {
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'avg_iou': metrics['avg_iou'],
                'TP': metrics['TP'],
                'FP': metrics['FP'],
                'FN': metrics['FN'],
            }
            result_entry['num_ground_truth'] = len(ground_truth)
        
        all_results[stem] = result_entry
    
    # Compute overall metrics
    if overall_metrics['TP'] + overall_metrics['FP'] > 0:
        overall_precision = overall_metrics['TP'] / (overall_metrics['TP'] + overall_metrics['FP'])
        overall_recall = overall_metrics['TP'] / max(1, overall_metrics['TP'] + overall_metrics['FN'])
        overall_f1 = 2 * overall_precision * overall_recall / max(1e-6, overall_precision + overall_recall)
        overall_avg_iou = overall_metrics['total_iou'] / max(1, overall_metrics['num_matched'])
        
        print(f"\n{'='*60}")
        print("OVERALL METRICS")
        print('='*60)
        print(f"Total Images: {len(all_results)}")
        print(f"True Positives: {overall_metrics['TP']}")
        print(f"False Positives: {overall_metrics['FP']}")
        print(f"False Negatives: {overall_metrics['FN']}")
        print(f"Precision: {overall_precision:.4f}")
        print(f"Recall: {overall_recall:.4f}")
        print(f"F1 Score: {overall_f1:.4f}")
        print(f"Average IoU: {overall_avg_iou:.4f}")
        
        # Add overall metrics to summary
        summary = {
            'overall_metrics': {
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1,
                'avg_iou': overall_avg_iou,
                'TP': overall_metrics['TP'],
                'FP': overall_metrics['FP'],
                'FN': overall_metrics['FN'],
                'num_images': len(all_results),
            },
            'per_image_results': all_results,
        }
    else:
        summary = {'per_image_results': all_results}
    
    # Save summary
    summary_path = output_dir / 'predictions_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved summary: {summary_path}")
    
    print(f"\n{'='*60}")
    print("Inference complete!")
    print('='*60)


if __name__ == '__main__':
    main()
