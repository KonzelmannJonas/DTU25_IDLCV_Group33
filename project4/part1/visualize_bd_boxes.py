import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
	from PIL import Image, ImageDraw, ImageFont
except ImportError as e:
	Image = None
	ImageDraw = None
	ImageFont = None


def parse_voc_xml(xml_path: Path):
	tree = ET.parse(str(xml_path))
	root = tree.getroot()

	filename = root.findtext("filename")
	size = root.find("size")
	width = int(size.findtext("width")) if size is not None else None
	height = int(size.findtext("height")) if size is not None else None

	objects = []
	for obj in root.findall("object"):
		name = obj.findtext("name") or "object"
		bnd = obj.find("bndbox")
		if bnd is None:
			continue
		xmin = int(float(bnd.findtext("xmin")))
		ymin = int(float(bnd.findtext("ymin")))
		xmax = int(float(bnd.findtext("xmax")))
		ymax = int(float(bnd.findtext("ymax")))
		objects.append({
			"name": name,
			"bbox": (xmin, ymin, xmax, ymax),
		})

	return {
		"filename": filename,
		"width": width,
		"height": height,
		"objects": objects,
	}


def draw_boxes(image_path: Path, objects, color=(255, 0, 0), thickness=3):
	im = Image.open(str(image_path)).convert("RGB")
	draw = ImageDraw.Draw(im)

	# Try to load a default font; fall back if not available
	try:
		font = ImageFont.load_default()
	except Exception:
		font = None

	for obj in objects:
		xmin, ymin, xmax, ymax = obj["bbox"]
		# Rectangle
		for t in range(thickness):
			draw.rectangle(
				[(xmin - t, ymin - t), (xmax + t, ymax + t)],
				outline=color,
				width=1,
			)
		# Label background and text
		label = obj.get("name", "object")
		if label and font is not None:
			# Measure text size in a Pillow 10+ compatible way
			try:
				# Preferred: use draw.textbbox if available
				bbox = draw.textbbox((0, 0), label, font=font)
				text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
			except AttributeError:
				# Fallbacks for older Pillow versions
				try:
					bbox = font.getbbox(label)
					text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
				except Exception:
					# Last resort: approximate size using getsize if present
					text_w, text_h = getattr(font, "getsize", lambda x: (50, 12))(label)

			pad = 2
			bg_box = [(xmin, max(0, ymin - text_h - 2 * pad)), (xmin + text_w + 2 * pad, ymin)]
			draw.rectangle(bg_box, fill=color)
			draw.text((xmin + pad, max(0, ymin - text_h - pad)), label, fill=(255, 255, 255), font=font)

	return im


def ensure_pillow():
	if Image is None:
		print("Error: Pillow is not installed. Install with: pip install pillow", file=sys.stderr)
		sys.exit(1)


def main():
	parser = argparse.ArgumentParser(description="Visualize bounding boxes from VOC XML annotations onto images.")
	parser.add_argument("--ann_dir", type=str, default=str(Path(__file__).parent / "data/examples/annotations"), help="Directory with VOC XML files")
	parser.add_argument("--img_dir", type=str, default=str(Path(__file__).parent / "data/examples/images"), help="Directory with images")
	parser.add_argument("--out_dir", type=str, default=str(Path(__file__).parent / "output"), help="Directory to save visualized images")
	parser.add_argument("--suffix", type=str, default="_boxed", help="Suffix to append to output filenames")
	parser.add_argument("--color", type=str, default="red", help="Box color: name or R,G,B (0-255)")
	parser.add_argument("--thickness", type=int, default=3, help="Box line thickness in pixels")
	parser.add_argument("--pattern", type=str, default="*.xml", help="Glob pattern for annotations (default: *.xml)")
	args = parser.parse_args()

	ensure_pillow()

	ann_dir = Path(args.ann_dir)
	img_dir = Path(args.img_dir)
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	# Parse color
	color = args.color
	if isinstance(color, str):
		if "," in color:
			try:
				r, g, b = map(int, color.split(","))
				color = (r, g, b)
			except Exception:
				color = (255, 0, 0)
		else:
			# Basic color names mapping
			name_map = {
				"red": (255, 0, 0),
				"green": (0, 200, 0),
				"blue": (0, 0, 255),
				"yellow": (255, 215, 0),
				"cyan": (0, 255, 255),
				"magenta": (255, 0, 255),
				"white": (255, 255, 255),
				"black": (0, 0, 0),
				"orange": (255, 140, 0),
			}
			color = name_map.get(color.lower(), (255, 0, 0))

	xml_files = sorted(ann_dir.glob(args.pattern))
	if not xml_files:
		print(f"No annotation files found in {ann_dir} matching {args.pattern}")
		return

	processed = 0
	missing_images = 0
	for xml_path in xml_files:
		data = parse_voc_xml(xml_path)
		filename = data.get("filename")
		if filename:
			img_path = img_dir / filename
		else:
			img_stem = xml_path.stem
			# Try common image extensions
			candidates = [img_dir / f"{img_stem}{ext}" for ext in [".jpg", ".jpeg", ".png", ".bmp"]]
			img_path = next((p for p in candidates if p.exists()), None)

		if img_path is None or not Path(img_path).exists():
			print(f"Warning: Image for {xml_path.name} not found in {img_dir}")
			missing_images += 1
			continue

		im = draw_boxes(Path(img_path), data["objects"], color=color, thickness=args.thickness)
		out_name = f"{Path(img_path).stem}{args.suffix}{Path(img_path).suffix}"
		out_path = out_dir / out_name
		im.save(str(out_path))
		processed += 1
		print(f"Saved: {out_path}")

	print(f"Done. Processed {processed} files. Missing images: {missing_images}.")


if __name__ == "__main__":
	main()

