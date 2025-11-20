#!/usr/bin/env python3
"""
Visualize labeled proposal boxes for training.

Reads per-image labeled proposal XML files (VOC-style) where each <object>
name is the class label (e.g., 'pothole' or 'background') and an optional
<score> contains IoU. Draws boxes with class-specific colors and optional
labels/scores, saving overlays for a directory of images.

Example:
python part1/visualize_labeled_proposals.py \
  --images-dir part1/data/examples/images \
  --labels-dir part1/output/labeled_proposals_xml \
  --out-dir part1/output/vis_labeled \
  --draw-pos --draw-neg --max-pos 200 --max-neg 200 --include-score
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import xml.etree.ElementTree as ET
import cv2

SUPPORTED_IMAGE_EXTS = {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
Box = Tuple[int,int,int,int]

class LabeledBox:
    def __init__(self, bbox: Box, label: str, score: float | None):
        self.bbox = bbox
        self.label = label
        self.score = score

def parse_labeled_xml(xml_path: Path) -> Tuple[str, int | None, int | None, List[LabeledBox]]:
    root = ET.parse(xml_path).getroot()
    filename = root.findtext('filename', default=xml_path.stem)
    width = root.find('size/width')
    height = root.find('size/height')
    W = int(width.text) if width is not None and width.text is not None else None
    H = int(height.text) if height is not None and height.text is not None else None
    boxes: List[LabeledBox] = []
    for obj in root.findall('object'):
        name = obj.findtext('name', default='object')
        score_text = obj.findtext('score')
        try:
            score = float(score_text) if score_text is not None else None
        except ValueError:
            score = None
        bb = obj.find('bndbox')
        if bb is None:
            continue
        xmin = int(float(bb.findtext('xmin', '0')))
        ymin = int(float(bb.findtext('ymin', '0')))
        xmax = int(float(bb.findtext('xmax', '0')))
        ymax = int(float(bb.findtext('ymax', '0')))
        boxes.append(LabeledBox((xmin,ymin,xmax,ymax), name, score))
    return filename, W, H, boxes

def draw_labeled_boxes(
    image_path: Path,
    labeled_boxes: List[LabeledBox],
    draw_pos: bool,
    draw_neg: bool,
    color_pos: Tuple[int,int,int],
    color_neg: Tuple[int,int,int],
    max_pos: int,
    max_neg: int,
    alpha: float,
    font_scale: float,
    font_thickness: int,
    include_score: bool,
) -> cv2.Mat:
    im = cv2.imread(str(image_path))
    if im is None:
        raise RuntimeError(f"Failed to read image {image_path}")
    overlay = im.copy()
    # Split by label
    positives = [lb for lb in labeled_boxes if lb.label.lower() != 'background'] if draw_pos else []
    negatives = [lb for lb in labeled_boxes if lb.label.lower() == 'background'] if draw_neg else []
    if max_pos > 0:
        positives = positives[:max_pos]
    if max_neg > 0:
        negatives = negatives[:max_neg]

    def draw(lb: LabeledBox, color: Tuple[int,int,int]):
        x1,y1,x2,y2 = lb.bbox
        cv2.rectangle(overlay, (x1,y1), (x2,y2), color, 2)
        label = lb.label
        if include_score and lb.score is not None:
            label = f"{label}:{lb.score:.2f}"
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            ty = max(0, y1 - 4)
            cv2.rectangle(overlay, (x1, ty - th - 4), (x1 + tw + 4, ty), color, -1)
            cv2.putText(overlay, label, (x1 + 2, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), font_thickness, cv2.LINE_AA)

    for lb in positives:
        draw(lb, color_pos)
    for lb in negatives:
        draw(lb, color_neg)

    if alpha < 1.0:
        im = cv2.addWeighted(overlay, alpha, im, 1-alpha, 0)
    else:
        im = overlay
    return im

def collect_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    label_files = {p.stem.replace('_labeled_proposals',''): p for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower()=='.xml'}
    for img in images_dir.iterdir():
        if img.is_file() and img.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            stem = img.stem
            if stem in label_files:
                pairs.append((img, label_files[stem]))
    return sorted(pairs)

def main():
    ap = argparse.ArgumentParser(description='Visualize labeled proposals for training.')
    ap.add_argument('--images-dir', required=True, help='Directory with images')
    ap.add_argument('--labels-dir', required=True, help='Directory with labeled proposal XML files')
    ap.add_argument('--out-dir', required=True, help='Directory to save overlay images')
    ap.add_argument('--draw-pos', action='store_true', help='Draw positive (non-background) boxes')
    ap.add_argument('--draw-neg', action='store_true', help='Draw negative (background) boxes')
    ap.add_argument('--max-pos', type=int, default=200, help='Max positives per image (0 = no limit)')
    ap.add_argument('--max-neg', type=int, default=200, help='Max negatives per image (0 = no limit)')
    ap.add_argument('--color-pos', default='0,255,0', help='B,G,R color for positives')
    ap.add_argument('--color-neg', default='0,0,255', help='B,G,R color for negatives')
    ap.add_argument('--alpha', type=float, default=1.0, help='Overlay transparency (1.0 solid)')
    ap.add_argument('--font-scale', type=float, default=0.5)
    ap.add_argument('--font-thickness', type=int, default=1)
    ap.add_argument('--include-score', action='store_true', help='Annotate IoU score when available')
    args = ap.parse_args()

    # Defaults: draw both if none specified
    draw_pos = args.draw_pos or (not args.draw_pos and not args.draw_neg)
    draw_neg = args.draw_neg or (not args.draw_pos and not args.draw_neg)

    def parse_color(s: str) -> Tuple[int,int,int]:
        parts = s.split(',')
        if len(parts) != 3:
            raise ValueError('Color must be B,G,R')
        return (int(parts[0]), int(parts[1]), int(parts[2]))

    color_pos = parse_color(args.color_pos)
    color_neg = parse_color(args.color_neg)

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(images_dir, labels_dir)
    if not pairs:
        print('No matching images and labeled XML files found.')
        return

    for img_path, xml_path in pairs:
        filename, W, H, boxes = parse_labeled_xml(xml_path)
        vis = draw_labeled_boxes(
            img_path,
            boxes,
            draw_pos,
            draw_neg,
            color_pos,
            color_neg,
            args.max_pos,
            args.max_neg,
            args.alpha,
            args.font_scale,
            args.font_thickness,
            args.include_score,
        )
        out_path = out_dir / f"{img_path.stem}_labeled_overlay.png"
        cv2.imwrite(str(out_path), vis)
        npos = sum(1 for b in boxes if b.label.lower() != 'background')
        nneg = sum(1 for b in boxes if b.label.lower() == 'background')
        print(f"Saved {out_path} (pos={npos}, neg={nneg})")

if __name__ == '__main__':
    main()
