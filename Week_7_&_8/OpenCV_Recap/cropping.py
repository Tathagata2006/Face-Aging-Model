import cv2

# Load the image
image = cv2.imread('sample.jpg')
h, w, _ = image.shape

# 1. Basic Cropping using NumPy Slicing 
# Syntax: image[start_y:end_y, start_x:end_x]
start_x, start_y = 100, 100
crop_w, crop_h = 200, 200
cropped_image = image[start_y:start_y+crop_h, start_x:start_x+crop_w]

# 2. Sequential Patch Generation 
patch_h, patch_w = 100, 100  # Dimensions for individual patches

for y in range(0, h, patch_h):
    for x in range(0, w, patch_w):
        # Edge case handling to prevent pulling out-of-bounds incomplete sections
        if y + patch_h > h or x + patch_w > w:
            continue
            
        # Slicing out the individual patch segments
        patch = image[y:y+patch_h, x:x+patch_w]
        
        # Save or process the generated patch
        cv2.imwrite(f'patch_{y}_{x}.jpg', patch)