import cv2
import numpy as np

# Load the image and find its spatial center
image = cv2.imread('sample.jpg')
h, w = image.shape[:2]
center = (w // 2, h // 2)

# 1. Image Rotation
# Define rotation parameters (center point, angle in degrees, scale factor)
angle = 45
scale = 1.0
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

# Apply the 2D transformation matrix using warpAffine
rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

# 2. Image Translation (Shifting)
# Define shift values along the X and Y axes
tx, ty = w // 4, h // 4

# Construct the affine transformation matrix for translation:
# [[1, 0, tx],
#  [0, 1, ty]]
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
translated_image = cv2.warpAffine(image, translation_matrix, (w, h))

# Save the transformations
cv2.imwrite('rotated.jpg', rotated_image)
cv2.imwrite('translated.jpg', translated_image)