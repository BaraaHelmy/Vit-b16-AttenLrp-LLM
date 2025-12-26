#!/usr/bin/env python3
"""Helper script to check the image used for LRP analysis."""

from pathlib import Path
from PIL import Image

# Read the saved image path
path_file = Path("lrp_image_path.txt")
if not path_file.exists():
    print("Error: lrp_image_path.txt not found. Run test_vit_forward.py first.")
    exit(1)

img_path = path_file.read_text().strip()
print(f"Image path: {img_path}")

# Check if file exists
if not Path(img_path).exists():
    print(f"Error: Image file not found at {img_path}")
    exit(1)

# Open and display image info
img = Image.open(img_path)
print(f"Image size: {img.size}")
print(f"Image mode: {img.mode}")
print(f"Image format: {img.format}")

