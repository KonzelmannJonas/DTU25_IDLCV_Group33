"""
Generate region proposals for all pothole images using Selective Search.
"""
import cv2
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import json

def generate_proposals_for_image(image_path, max_proposals=200):
    """Generate proposals for a single image using Selective Search."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not load {image_path}")
        return []
    
    # Initialize Selective Search
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()  # Fast mode
    
    # Get proposals
    rects = ss.process()
    
    # Limit number of proposals
    proposals = []
    for i, (x, y, w, h) in enumerate(rects[:max_proposals]):
        proposals.append({
            'x': int(x),
            'y': int(y),
            'width': int(w),
            'height': int(h)
        })
    
    return proposals

def save_proposals_to_xml(proposals, output_path, image_name, img_width, img_height):
    """Save proposals to XML file in Pascal VOC format."""
    root = ET.Element('annotation')
    
    # Add image info
    folder = ET.SubElement(root, 'folder')
    folder.text = 'proposals'
    
    filename = ET.SubElement(root, 'filename')
    filename.text = f"{image_name}.png"
    
    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(img_width)
    height = ET.SubElement(size, 'height')
    height.text = str(img_height)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
    
    # Add proposals
    for prop in proposals:
        obj = ET.SubElement(root, 'object')
        
        name = ET.SubElement(obj, 'name')
        name.text = 'proposal'
        
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(prop['x'])
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(prop['y'])
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(prop['x'] + prop['width'])
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(prop['y'] + prop['height'])
    
    # Write to file
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding='utf-8', xml_declaration=True)

def main():
    # Configuration
    image_dir = "/dtu/datasets1/02516/potholes/images"
    output_dir = "proposals"
    max_proposals = 200
    
    # Load splits
    with open('splits.json', 'r') as f:
        splits = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all images (train + test)
    all_images = splits['train'] + splits['test']
    
    print(f"Generating proposals for {len(all_images)} images...")
    
    for image_name in tqdm(all_images):
        image_path = os.path.join(image_dir, f"{image_name}.png")
        output_path = os.path.join(output_dir, f"{image_name}_proposals.xml")
        
        # Skip if already exists
        if os.path.exists(output_path):
            continue
        
        # Load image to get dimensions
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not load {image_path}")
            continue
        
        img_height, img_width = img.shape[:2]
        
        # Generate proposals
        proposals = generate_proposals_for_image(image_path, max_proposals)
        
        # Save to XML
        save_proposals_to_xml(proposals, output_path, image_name, img_width, img_height)
    
    print(f"Done! Proposals saved to {output_dir}/")

if __name__ == "__main__":
    main()
