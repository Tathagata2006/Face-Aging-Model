import cv2
import numpy as np

# Load the input image
image = cv2.imread('sample.jpg')

# 1. Resize using specific pixel values (width, height)
# Note: This can alter the original aspect ratio
target_size = (300, 200) 
resized_pixels = cv2.resize(image, target_size)

# 2. Resize using scaling factors (e.g., shrinking by half)
# Setting dsize to None allows the fx and fy factors to dictate the new size
resized_scaled = cv2.resize(image, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

# Save or display your results
cv2.imwrite('resized_pixels.jpg', resized_pixels)
cv2.imwrite('resized_scaled.jpg', resized_scaled)