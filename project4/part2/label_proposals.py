"""
Label proposals based on IoU with ground truth bounding boxes.
"""
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import json

def parse_xml_boxes(xml_path):
    """Parse bounding boxes from XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        
        box = {
            'name': name,
            'xmin': int(bndbox.find('xmin').text),
            'ymin': int(bndbox.find('ymin').text),
            'xmax': int(bndbox.find('xmax').text),
            'ymax': int(bndbox.find('ymax').text)
        }
        boxes.append(box)
    
    return boxes

def compute_iou(box1, box2):
    """Compute IoU between two boxes."""
    # Calculate intersection
    x1 = max(box1['xmin'], box2['xmin'])
    y1 = max(box1['ymin'], box2['ymin'])
    x2 = min(box1['xmax'], box2['xmax'])
    y2 = min(box1['ymax'], box2['ymax'])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
    area2 = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def label_proposals(proposals_path, annotations_path, output_path, iou_threshold=0.5):
    """Label proposals based on IoU with ground truth."""
    # Parse proposals
    proposals = parse_xml_boxes(proposals_path)
    
    # Parse ground truth
    gt_boxes = parse_xml_boxes(annotations_path)
    
    # Label each proposal
    labeled_proposals = []
    for prop in proposals:
        max_iou = 0.0
        best_gt_name = 'background'
        
        # Find best matching ground truth box
        for gt in gt_boxes:
            iou = compute_iou(prop, gt)
            if iou > max_iou:
                max_iou = iou
                best_gt_name = gt['name']
        
        # Assign label based on IoU threshold
        label = best_gt_name if max_iou >= iou_threshold else 'background'
        
        labeled_proposals.append({
            'box': prop,
            'label': label,
            'iou': max_iou
        })
    
    # Save labeled proposals to XML
    save_labeled_proposals_xml(labeled_proposals, output_path, proposals_path)
    
    return labeled_proposals

def save_labeled_proposals_xml(labeled_proposals, output_path, original_proposals_path):
    """Save labeled proposals to XML file."""
    # Parse original to get metadata
    tree = ET.parse(original_proposals_path)
    root = tree.getroot()
    
    # Create new root
    new_root = ET.Element('annotation')
    
    # Copy folder, filename, size
    for tag in ['folder', 'filename', 'size']:
        elem = root.find(tag)
        if elem is not None:
            new_root.append(elem)
    
    # Add labeled proposals
    for lp in labeled_proposals:
        obj = ET.SubElement(new_root, 'object')
        
        name = ET.SubElement(obj, 'name')
        name.text = lp['label']
        
        iou_elem = ET.SubElement(obj, 'iou')
        iou_elem.text = f"{lp['iou']:.4f}"
        
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(lp['box']['xmin'])
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(lp['box']['ymin'])
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(lp['box']['xmax'])
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(lp['box']['ymax'])
    
    # Write to file
    tree = ET.ElementTree(new_root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding='utf-8', xml_declaration=True)

def main():
    # Configuration
    proposals_dir = "proposals"
    annotations_dir = "/dtu/datasets1/02516/potholes/annotations"
    output_dir = "labeled_proposals"
    iou_threshold = 0.5
    
    # Load splits
    with open('splits.json', 'r') as f:
        splits = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all images
    all_images = splits['train'] + splits['test']
    
    print(f"Labeling proposals for {len(all_images)} images...")
    
    for image_name in tqdm(all_images):
        proposals_path = os.path.join(proposals_dir, f"{image_name}_proposals.xml")
        annotations_path = os.path.join(annotations_dir, f"{image_name}.xml")
        output_path = os.path.join(output_dir, f"{image_name}_labeled_proposals.xml")
        
        # Skip if already exists
        if os.path.exists(output_path):
            continue
        
        # Check if files exist
        if not os.path.exists(proposals_path):
            print(f"Warning: Proposals not found for {image_name}")
            continue
        
        if not os.path.exists(annotations_path):
            print(f"Warning: Annotations not found for {image_name}")
            continue
        
        # Label proposals
        label_proposals(proposals_path, annotations_path, output_path, iou_threshold)
    
    print(f"Done! Labeled proposals saved to {output_dir}/")

if __name__ == "__main__":
    main()
