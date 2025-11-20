#!/usr/bin/env python3
"""
Generate object proposals for a set of images using Selective Search (and optionally EdgeBoxes).
Outputs Pascal VOC-style XML files listing proposal boxes (one <object> per proposal). Evaluation metrics are printed to stdout.

Usage examples:

Selective Search fast mode, limit to 1500 proposals:
    python generate_proposals.py \
        --images-dir part1/data/examples/images \
        --ann-dir part1/data/examples/annotations \
        --method selective_search --ss-mode fast \
        --max-proposals 1500 \
        --output-dir part1/output/proposals

EdgeBoxes (requires structured edge model file model.yml.gz):
    python generate_proposals.py --method edge_boxes --edge-model /path/to/model.yml.gz ...

"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import xml.etree.ElementTree as ET

import cv2
import numpy as np

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def list_images(images_dir: Path) -> List[Path]:
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTS])

def parse_voc_xml(xml_path: Path) -> Dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size_node = root.find('size')
    width = int(size_node.find('width').text) if size_node is not None else None
    height = int(size_node.find('height').text) if size_node is not None else None
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text if obj.find('name') is not None else 'object'
        bnd = obj.find('bndbox')
        if bnd is None:
            continue
        xmin = int(float(bnd.find('xmin').text))
        ymin = int(float(bnd.find('ymin').text))
        xmax = int(float(bnd.find('xmax').text))
        ymax = int(float(bnd.find('ymax').text))
        objects.append({"label": name, "bbox": [xmin, ymin, xmax, ymax]})
    filename_node = root.find('filename')
    filename = filename_node.text if filename_node is not None else xml_path.stem
    return {"filename": filename, "width": width, "height": height, "objects": objects}

def resize_image_keep_aspect(im: np.ndarray, target_longer: int) -> Tuple[np.ndarray, float]:
    if target_longer <= 0:
        return im, 1.0
    h, w = im.shape[:2]
    longer = max(h, w)
    if longer <= target_longer:
        return im, 1.0
    scale = target_longer / float(longer)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale

def selective_search(im: np.ndarray, mode: str = 'fast', max_proposals: int = 2000) -> List[Tuple[int,int,int,int]]:
    """Run Selective Search using OpenCV contrib if available, else fall back to the
    pure-Python 'selectivesearch' package.

    Parameters
    ----------
    im : np.ndarray (BGR)
        Input image.
    mode : {'fast','quality'}
        Trade-off between speed and proposal diversity. In fallback implementation
        this controls segmentation scale and min_size.
    max_proposals : int
        Limit number of returned boxes.
    """
    # Try OpenCV contrib implementation first
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'segmentation'):
        try:
            ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            ss.setBaseImage(im)
            if mode == 'quality':
                ss.switchToSelectiveSearchQuality()
            else:
                ss.switchToSelectiveSearchFast()
            rects = ss.process()  # list of (x,y,w,h)
            boxes = []
            for (x, y, w, h) in rects[:max_proposals]:
                boxes.append((x, y, x + w, y + h))
            return boxes
        except Exception:
            # Fall through to python implementation
            pass
    # Fallback strategies when neither contrib nor selectivesearch package is available.
    try:
        import selectivesearch  # type: ignore
        img_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if mode == 'quality':
            scale = 450; min_size = 20
        else:
            scale = 150; min_size = 50
        _, regions = selectivesearch.selective_search(img_rgb, scale=scale, sigma=0.8, min_size=min_size)
        boxes = []
        seen = set()
        for r in regions:
            x, y, w, h = r['rect']
            box = (x, y, x + w, y + h)
            if box in seen:
                continue
            seen.add(box)
            boxes.append(box)
            if len(boxes) >= max_proposals:
                break
        return boxes
    except ImportError:
        pass

    # Pure OpenCV fallback (approximate proposals): contours + MSER + multi-scale grid + random boxes.
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Mean-shift filtering to smooth colors (helps contour quality)
    try:
        filtered = cv2.pyrMeanShiftFiltering(im, 20, 45)
    except Exception:
        filtered = im
    gray_f = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_f, 60, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = gray.shape
    boxes_set = set()
    boxes_list: List[Tuple[int,int,int,int]] = []

    def add_box(x1, y1, x2, y2):
        x1 = max(0, min(w_img-1, x1)); x2 = max(0, min(w_img-1, x2))
        y1 = max(0, min(h_img-1, y1)); y2 = max(0, min(h_img-1, y2))
        if x2 <= x1 or y2 <= y1:
            return
        box = (int(x1), int(y1), int(x2), int(y2))
        if box not in boxes_set:
            boxes_set.add(box)
            boxes_list.append(box)

    # Contour boxes
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 200 or w < 10 or h < 10:
            continue
        if area > 0.5 * w_img * h_img:  # skip overly large
            continue
        add_box(x, y, x + w, y + h)
        if len(boxes_list) >= max_proposals:
            return boxes_list[:max_proposals]

    # MSER regions if available
    if hasattr(cv2, 'MSER_create'):
        try:
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)
            for pts in regions:
                x, y, w, h = cv2.boundingRect(pts)
                area = w * h
                if area < 150 or w < 8 or h < 8:
                    continue
                add_box(x, y, x + w, y + h)
                if len(boxes_list) >= max_proposals:
                    return boxes_list[:max_proposals]
        except Exception:
            pass

    # Multi-scale sliding windows (coarse)
    scales = [0.25, 0.5, 0.75] if mode == 'quality' else [0.4, 0.7]
    for s in scales:
        win_w = int(w_img * s)
        win_h = int(h_img * s)
        step_x = max(16, win_w // 4)
        step_y = max(16, win_h // 4)
        for y in range(0, h_img - win_h + 1, step_y):
            for x in range(0, w_img - win_w + 1, step_x):
                add_box(x, y, x + win_w, y + win_h)
                if len(boxes_list) >= max_proposals:
                    return boxes_list[:max_proposals]

    # Random jitter boxes for diversity
    rng = np.random.default_rng(42)
    num_rand = 300 if mode == 'quality' else 150
    for _ in range(num_rand):
        rw = rng.integers(low=int(0.05*w_img), high=int(0.5*w_img))
        rh = rng.integers(low=int(0.05*h_img), high=int(0.5*h_img))
        rx = rng.integers(0, w_img - rw)
        ry = rng.integers(0, h_img - rh)
        add_box(int(rx), int(ry), int(rx + rw), int(ry + rh))
        if len(boxes_list) >= max_proposals:
            break

    return boxes_list[:max_proposals]

def edge_boxes(im: np.ndarray, model_path: Path, max_boxes: int = 2000) -> List[Tuple[int,int,int,int]]:
    # EdgeBoxes requires structured edge model.
    if not model_path.exists():
        raise FileNotFoundError(f"Edge model not found: {model_path}")
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    edge_detector = cv2.ximgproc.createStructuredEdgeDetection(str(model_path))
    edges = edge_detector.detectEdges(np.float32(rgb) / 255.0)
    orientation = edge_detector.computeOrientation(edges)
    edges_nms = cv2.ximgproc.edgesNms(edges, orientation)
    eb = cv2.ximgproc.createEdgeBoxes()
    eb.setMaxBoxes(max_boxes)
    boxes, scores = eb.getBoundingBoxes(edges_nms, orientation)
    out = []
    for (x, y, w, h), s in zip(boxes, scores):
        out.append((x, y, x + w, y + h))
    return out

def compute_iou(boxA: Tuple[int,int,int,int], boxB: Tuple[int,int,int,int]) -> float:
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

def evaluate(boxes: List[Tuple[int,int,int,int]], gt_objs: List[Dict], iou_thresh: float = 0.5) -> Dict:
    gt_boxes = [tuple(obj['bbox']) for obj in gt_objs]
    hits = 0
    ious = []
    for gt in gt_boxes:
        best = 0.0
        for b in boxes:
            iou = compute_iou(b, gt)
            if iou > best:
                best = iou
        ious.append(best)
        if best >= iou_thresh:
            hits += 1
    recall = hits / len(gt_boxes) if gt_boxes else None
    mean_iou = float(np.mean(ious)) if ious else None
    return {"recall@{:.2f}".format(iou_thresh): recall, "mean_best_iou": mean_iou}

def write_voc_xml(
    out_path: Path,
    image_filename: str,
    width: int,
    height: int,
    boxes: List[Tuple[int,int,int,int]],
    label_prefix: str = "proposal",
) -> None:
    """Write proposals to a Pascal VOC-style XML file (one <object> per box)."""
    ann = ET.Element('annotation')
    ET.SubElement(ann, 'filename').text = image_filename
    size_el = ET.SubElement(ann, 'size')
    ET.SubElement(size_el, 'width').text = str(int(width))
    ET.SubElement(size_el, 'height').text = str(int(height))
    ET.SubElement(size_el, 'depth').text = '3'
    ET.SubElement(ann, 'segmented').text = '0'
    for (x1,y1,x2,y2) in boxes:
        obj_el = ET.SubElement(ann, 'object')
        ET.SubElement(obj_el, 'name').text = label_prefix
        ET.SubElement(obj_el, 'pose').text = 'Unspecified'
        ET.SubElement(obj_el, 'truncated').text = '0'
        ET.SubElement(obj_el, 'difficult').text = '0'
        bb_el = ET.SubElement(obj_el, 'bndbox')
        ET.SubElement(bb_el, 'xmin').text = str(int(x1))
        ET.SubElement(bb_el, 'ymin').text = str(int(y1))
        ET.SubElement(bb_el, 'xmax').text = str(int(x2))
        ET.SubElement(bb_el, 'ymax').text = str(int(y2))
    tree = ET.ElementTree(ann)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out_path), encoding='utf-8', xml_declaration=True)

def run_on_image(path: Path, args) -> Dict:
    im = cv2.imread(str(path))
    if im is None:
        raise RuntimeError(f"Failed to read image {path}")
    original_shape = im.shape[:2]
    im_resized, scale = resize_image_keep_aspect(im, args.resize_longer_side)
    if scale != 1.0:
        # Note: proposals are generated on resized image, will be mapped back.
        pass
    if args.method == 'selective_search':
        boxes = selective_search(im_resized, mode=args.ss_mode, max_proposals=args.max_proposals)
    elif args.method == 'edge_boxes':
        if args.edge_model is None:
            raise ValueError("EdgeBoxes requires --edge-model path to model.yml.gz")
        boxes = edge_boxes(im_resized, Path(args.edge_model), max_boxes=args.max_proposals)
    else:
        raise ValueError(f"Unknown method {args.method}")
    # Map boxes back if resized
    if scale != 1.0:
        inv = 1.0 / scale
        boxes = [(int(round(x1 * inv)), int(round(y1 * inv)), int(round(x2 * inv)), int(round(y2 * inv))) for (x1,y1,x2,y2) in boxes]
    record = {
        "image": path.name,
        "original_height": original_shape[0],
        "original_width": original_shape[1],
        "num_boxes": len(boxes),
        "boxes": boxes,
    }
    return record

def main():
    parser = argparse.ArgumentParser(description="Generate object proposals (Selective Search / EdgeBoxes)")
    parser.add_argument('--images-dir', required=True, help='Directory with input images')
    parser.add_argument('--ann-dir', help='Directory with VOC XML annotations for evaluation')
    parser.add_argument('--method', choices=['selective_search', 'edge_boxes'], default='selective_search')
    parser.add_argument('--ss-mode', choices=['fast', 'quality'], default='fast', help='Selective Search speed/quality trade-off')
    parser.add_argument('--edge-model', help='Path to model.yml.gz for EdgeBoxes structured edge detection')
    parser.add_argument('--resize-longer-side', type=int, default=0, help='Resize longer side to this (0=no resize)')
    parser.add_argument('--max-proposals', type=int, default=200)
    parser.add_argument('--iou-thresh', type=float, default=0.5, help='IoU threshold for recall metric')
    parser.add_argument('--output-dir', required=True, help='Directory to write XML proposal files and optional visuals')
    parser.add_argument('--visualize', action='store_true', help='Save a visualization for first N boxes (50) per image')
    parser.add_argument('--vis-max', type=int, default=50, help='Max boxes to draw when visualizing')
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    ann_dir = Path(args.ann_dir) if args.ann_dir else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(images_dir)
    if not images:
        print(f"No images found in {images_dir}", file=sys.stderr)
        sys.exit(1)

    ann_map = {}
    if ann_dir and ann_dir.exists():
        for xml_path in sorted(ann_dir.glob('*.xml')):
            data = parse_voc_xml(xml_path)
            ann_map[data['filename']] = data
    else:
        if ann_dir:
            print(f"Annotation directory {ann_dir} does not exist; skipping evaluation", file=sys.stderr)

    all_metrics = []
    for img_path in images:
        rec = run_on_image(img_path, args)
        # Evaluation
        metrics = None
        if ann_map.get(img_path.name):
            gt_objs = ann_map[img_path.name]['objects']
            metrics = evaluate(rec['boxes'], gt_objs, iou_thresh=args.iou_thresh)
            rec['metrics'] = metrics
            all_metrics.append(metrics)
        # Write per-image XML
        xml_path = out_dir / f"{img_path.stem}_proposals.xml"
        write_voc_xml(
            xml_path,
            image_filename=rec['image'],
            width=rec['original_width'],
            height=rec['original_height'],
            boxes=rec['boxes'],
            label_prefix='proposal',
        )
        # Optional visualization
        if args.visualize:
            vis_im = cv2.imread(str(img_path))
            for i, (x1,y1,x2,y2) in enumerate(rec['boxes'][:args.vis_max]):
                cv2.rectangle(vis_im, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_vis.png"), vis_im)
        print(f"Processed {img_path.name}: {rec['num_boxes']} boxes -> {xml_path.name}")

    # Aggregate metrics (print only; no JSON file)
    if all_metrics:
        recalls = [m[f"recall@{args.iou_thresh:.2f}"] for m in all_metrics if m.get(f"recall@{args.iou_thresh:.2f}") is not None]
        mean_ious = [m['mean_best_iou'] for m in all_metrics if m.get('mean_best_iou') is not None]
        summary = {
            'num_images_evaluated': len(all_metrics),
            f'avg_recall@{args.iou_thresh:.2f}': float(np.mean(recalls)) if recalls else None,
            'avg_mean_best_iou': float(np.mean(mean_ious)) if mean_ious else None,
        }
        print("Evaluation summary:", summary)

if __name__ == '__main__':
    main()
