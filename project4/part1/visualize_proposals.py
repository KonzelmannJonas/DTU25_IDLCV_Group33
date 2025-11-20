#!/usr/bin/env python3
"""
Visualize bounding box proposals on images.
Supports proposals stored either as:
  - Pascal VOC style XML (each <object> has a <bndbox>) produced by generate_proposals.py (--output-format xml/both)
  - JSON proposal files produced by generate_proposals.py (contains key "boxes")

Examples:
Single image + XML proposals:
  python visualize_proposals.py \
    --image part1/data/examples/images/potholes2.png \
    --proposals part1/output/proposals_xml/potholes2_proposals.xml \
    --out part1/output/proposals_xml/potholes2_overlay.png --max 100

Batch (match filenames by stem):
  python visualize_proposals.py \
    --images-dir part1/data/examples/images \
    --proposals-dir part1/output/proposals_xml \
    --out-dir part1/output/vis_proposals --max 50

"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional
import cv2

Box = Tuple[int,int,int,int]

def load_boxes_from_xml(xml_path: Path) -> List[Box]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes: List[Box] = []
    for obj in root.findall('object'):
        bb = obj.find('bndbox')
        if bb is None:
            continue
        xmin = int(float(bb.find('xmin').text))
        ymin = int(float(bb.find('ymin').text))
        xmax = int(float(bb.find('xmax').text))
        ymax = int(float(bb.find('ymax').text))
        boxes.append((xmin, ymin, xmax, ymax))
    return boxes

def load_boxes_from_json(json_path: Path) -> List[Box]:
    with open(json_path, 'r') as f:
        data = json.load(f)
    boxes_raw = data.get('boxes', [])
    boxes: List[Box] = []
    for b in boxes_raw:
        # b may be list [x1,y1,x2,y2]
        if len(b) == 4:
            boxes.append((int(b[0]), int(b[1]), int(b[2]), int(b[3])))
    return boxes

def auto_load_proposals(path: Path) -> List[Box]:
    if path.suffix.lower() == '.xml':
        return load_boxes_from_xml(path)
    elif path.suffix.lower() == '.json':
        return load_boxes_from_json(path)
    else:
        raise ValueError(f"Unsupported proposals file extension: {path.suffix}")

def draw_boxes(im_path: Path, boxes: List[Box], max_boxes: int, color: Tuple[int,int,int], thickness: int, alpha: float, enumerate_idx: bool, font_scale: float, font_thickness: int) -> cv2.Mat:
    im = cv2.imread(str(im_path))
    if im is None:
        raise RuntimeError(f"Failed to read image {im_path}")
    overlay = im.copy()
    for i, (x1,y1,x2,y2) in enumerate(boxes[:max_boxes]):
        cv2.rectangle(overlay, (x1,y1), (x2,y2), color, thickness)
        if enumerate_idx:
            label = str(i)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            # Background rectangle for text
            cv2.rectangle(overlay, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(overlay, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), font_thickness, cv2.LINE_AA)
    if alpha < 1.0:
        im = cv2.addWeighted(overlay, alpha, im, 1-alpha, 0)
    else:
        im = overlay
    return im

def collect_pairs(images_dir: Path, proposals_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    proposal_files = {p.stem.replace('_proposals',''): p for p in proposals_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.xml', '.json'}}
    for img in images_dir.iterdir():
        if img.is_file() and img.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}:
            stem = img.stem
            if stem in proposal_files:
                pairs.append((img, proposal_files[stem]))
    return sorted(pairs)

def main():
    ap = argparse.ArgumentParser(description='Visualize bounding box proposals.')
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', help='Single image path')
    group.add_argument('--images-dir', help='Directory of images (batch mode)')
    ap.add_argument('--proposals', help='Single proposals file (.xml or .json)')
    ap.add_argument('--proposals-dir', help='Directory containing proposal files (batch mode)')
    ap.add_argument('--out', help='Output image path (single mode)')
    ap.add_argument('--out-dir', help='Output dir for batch mode')
    ap.add_argument('--max', type=int, default=100, help='Maximum number of boxes to draw')
    ap.add_argument('--color', default='0,255,0', help='Box color B,G,R (default 0,255,0)')
    ap.add_argument('--thickness', type=int, default=2, help='Box line thickness')
    ap.add_argument('--alpha', type=float, default=1.0, help='Overlay transparency (1.0 = solid)')
    ap.add_argument('--enumerate', action='store_true', help='Draw box indices')
    ap.add_argument('--font-scale', type=float, default=0.5)
    ap.add_argument('--font-thickness', type=int, default=1)
    args = ap.parse_args()

    color_parts = [int(c) for c in args.color.split(',')]
    if len(color_parts) != 3:
        raise ValueError('Color must have 3 comma-separated ints B,G,R')
    color = tuple(color_parts)  # type: ignore

    if args.image:
        if not args.proposals:
            raise ValueError('Provide --proposals for single image mode')
        boxes = auto_load_proposals(Path(args.proposals))
        out_path = Path(args.out) if args.out else Path(f"{Path(args.image).stem}_proposals_overlay.png")
        vis = draw_boxes(Path(args.image), boxes, args.max, color, args.thickness, args.alpha, args.enumerate, args.font_scale, args.font_thickness)
        cv2.imwrite(str(out_path), vis)
        print(f"Wrote visualization: {out_path} ({len(boxes)} boxes, showing {min(len(boxes), args.max)})")
    else:
        if not args.images_dir or not args.proposals_dir:
            raise ValueError('Batch mode requires --images-dir and --proposals-dir')
        out_dir = Path(args.out_dir) if args.out_dir else Path('proposal_visualizations')
        out_dir.mkdir(parents=True, exist_ok=True)
        pairs = collect_pairs(Path(args.images_dir), Path(args.proposals_dir))
        if not pairs:
            print('No matching image/proposal pairs found.', flush=True)
        for img_path, prop_path in pairs:
            boxes = auto_load_proposals(prop_path)
            vis = draw_boxes(img_path, boxes, args.max, color, args.thickness, args.alpha, args.enumerate, args.font_scale, args.font_thickness)
            out_path = out_dir / f"{img_path.stem}_overlay.png"
            cv2.imwrite(str(out_path), vis)
            print(f"Saved {out_path} ({len(boxes)} boxes, showing {min(len(boxes), args.max)})")

if __name__ == '__main__':
    main()
