#!/usr/bin/env python3
"""
Assign labels to proposal boxes by matching them to ground-truth annotations via IoU
and save labeled proposals as Pascal VOC-style XML files (one <object> per proposal).

- Positives: IoU >= --pos-iou get the GT class label (e.g., 'pothole')
- Negatives: IoU < --neg-iou labeled as 'background'
- Between thresholds: ignored (not written)
- Optional balancing: cap negatives to N per positive

Example:
python part1/label_proposals.py \
  --proposals-dir part1/output/proposals_xml \
  --ann-dir part1/data/examples/annotations \
  --splits part1/data/examples/splits.json --split train \
  --pos-iou 0.5 --neg-iou 0.3 --max-neg-per-pos 5 \
  --output-dir part1/output/labeled_proposals_xml
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

Box = Tuple[int,int,int,int]

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def parse_voc_xml(xml_path: Path) -> Dict:
    root = ET.parse(xml_path).getroot()
    size_node = root.find('size')
    width = int(size_node.find('width').text) if size_node is not None else None
    height = int(size_node.find('height').text) if size_node is not None else None
    filename = (root.find('filename').text if root.find('filename') is not None else xml_path.stem)
    objects = []
    for obj in root.findall('object'):
        name_node = obj.find('name')
        bb = obj.find('bndbox')
        if bb is None:
            continue
        xmin = int(float(bb.find('xmin').text)); ymin = int(float(bb.find('ymin').text))
        xmax = int(float(bb.find('xmax').text)); ymax = int(float(bb.find('ymax').text))
        objects.append({"label": name_node.text if name_node is not None else "object", "bbox": [xmin,ymin,xmax,ymax]})
    return {"filename": filename, "width": width, "height": height, "objects": objects}

def load_proposals_xml(xml_path: Path) -> List[Box]:
    root = ET.parse(xml_path).getroot()
    boxes: List[Box] = []
    for obj in root.findall('object'):
        bb = obj.find('bndbox')
        if bb is None:
            continue
        xmin = int(float(bb.find('xmin').text)); ymin = int(float(bb.find('ymin').text))
        xmax = int(float(bb.find('xmax').text)); ymax = int(float(bb.find('ymax').text))
        boxes.append((xmin,ymin,xmax,ymax))
    return boxes

def iou(a: Box, b: Box) -> float:
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    iw = max(0, xB - xA); ih = max(0, yB - yA)
    inter = iw * ih
    areaA = (a[2]-a[0]) * (a[3]-a[1])
    areaB = (b[2]-b[0]) * (b[3]-b[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

def write_labeled_voc_xml(out_path: Path, image_filename: str, width: int, height: int, labeled_boxes: List[Tuple[Box, str, float]]) -> None:
    ann = ET.Element('annotation')
    ET.SubElement(ann, 'filename').text = image_filename
    size_el = ET.SubElement(ann, 'size')
    ET.SubElement(size_el, 'width').text = str(int(width) if width is not None else 0)
    ET.SubElement(size_el, 'height').text = str(int(height) if height is not None else 0)
    ET.SubElement(size_el, 'depth').text = '3'
    ET.SubElement(ann, 'segmented').text = '0'
    for (x1,y1,x2,y2), label, best_iou in labeled_boxes:
        obj_el = ET.SubElement(ann, 'object')
        ET.SubElement(obj_el, 'name').text = label
        ET.SubElement(obj_el, 'pose').text = 'Unspecified'
        ET.SubElement(obj_el, 'truncated').text = '0'
        ET.SubElement(obj_el, 'difficult').text = '0'
        # Optionally store IoU as a custom field
        ET.SubElement(obj_el, 'score').text = f"{best_iou:.4f}"
        bb_el = ET.SubElement(obj_el, 'bndbox')
        ET.SubElement(bb_el, 'xmin').text = str(int(x1))
        ET.SubElement(bb_el, 'ymin').text = str(int(y1))
        ET.SubElement(bb_el, 'xmax').text = str(int(x2))
        ET.SubElement(bb_el, 'ymax').text = str(int(y2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(ann).write(str(out_path), encoding='utf-8', xml_declaration=True)

def main():
    ap = argparse.ArgumentParser(description='Label proposals using IoU with ground-truth and save as VOC XML.')
    ap.add_argument('--proposals-dir', required=True, help='Directory with per-image proposals XML files')
    ap.add_argument('--ann-dir', required=True, help='Directory with ground-truth VOC XML annotations')
    ap.add_argument('--splits', help='Path to splits.json containing {"train": [...], "test": [...]}')
    ap.add_argument('--split', choices=['train','test','all'], default='train', help='Which split to process')
    ap.add_argument('--pos-iou', type=float, default=0.5, help='IoU threshold for positive label')
    ap.add_argument('--neg-iou', type=float, default=0.3, help='IoU threshold for background label')
    ap.add_argument('--max-neg-per-pos', type=int, default=5, help='Cap negatives per positive (set 0 to disable capping)')
    ap.add_argument('--output-dir', required=True, help='Output directory for labeled XML files')
    args = ap.parse_args()

    proposals_dir = Path(args.proposals_dir)
    ann_dir = Path(args.ann_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which images to process
    stems: List[str]
    if args.splits and args.split != 'all':
        with open(args.splits, 'r') as f:
            splits = json.load(f)
        stems = list(splits[args.split])
    else:
        stems = sorted([p.stem.replace('_proposals','') for p in proposals_dir.glob('*_proposals.xml')])

    total_pos = 0
    total_neg = 0
    for stem in stems:
        prop_xml = proposals_dir / f"{stem}_proposals.xml"
        gt_xml = ann_dir / f"{stem}.xml"
        if not prop_xml.exists() or not gt_xml.exists():
            # Skip if either proposals or GT is missing
            continue
        props = load_proposals_xml(prop_xml)
        gt = parse_voc_xml(gt_xml)
        gt_boxes = [tuple(obj['bbox']) for obj in gt['objects']]
        gt_labels = [obj['label'] for obj in gt['objects']]

        labeled: List[Tuple[Box, str, float]] = []
        pos_count = 0
        neg_count = 0
        for p in props:
            best = 0.0
            best_label = None
            for gbox, glabel in zip(gt_boxes, gt_labels):
                val = iou(p, gbox)
                if val > best:
                    best = val
                    best_label = glabel
            if best >= args.pos_iou and best_label is not None:
                labeled.append((p, best_label, best))
                pos_count += 1
            elif best < args.neg_iou:
                # Background; apply optional cap later
                labeled.append((p, 'background', best))
                neg_count += 1
            else:
                # ignore ambiguous
                pass
        # Balance negatives
        if args.max_neg_per_pos > 0 and pos_count > 0:
            max_neg = args.max_neg_per_pos * pos_count
            # Keep positives and trim backgrounds to max_neg
            positives = [x for x in labeled if x[1] != 'background']
            negatives = [x for x in labeled if x[1] == 'background']
            negatives = negatives[:max_neg]
            labeled = positives + negatives
            neg_count = len(negatives)
            pos_count = len(positives)

        # Write labeled XML
        out_xml = out_dir / f"{stem}_labeled_proposals.xml"
        width = gt.get('width', 0)
        height = gt.get('height', 0)
        image_filename = gt.get('filename', f"{stem}.png")
        write_labeled_voc_xml(out_xml, image_filename=image_filename, width=width or 0, height=height or 0, labeled_boxes=labeled)
        total_pos += pos_count
        total_neg += neg_count
        print(f"{stem}: wrote {out_xml.name} (pos={pos_count}, neg={neg_count})")

    print(f"Done. Total pos={total_pos}, neg={total_neg} across {len(stems)} images.")

if __name__ == '__main__':
    main()
