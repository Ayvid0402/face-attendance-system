# check_image.py - diagnose the image format
from PIL import Image
import numpy as np

photo_path = "known_faces/divya_1.jpeg"

# open with PIL
img = Image.open(photo_path)

print(f"Format   : {img.format}")
print(f"Mode     : {img.mode}")
print(f"Size     : {img.size}")

# convert to numpy and check
img_array = np.array(img)
print(f"Shape    : {img_array.shape}")
print(f"Dtype    : {img_array.dtype}")
