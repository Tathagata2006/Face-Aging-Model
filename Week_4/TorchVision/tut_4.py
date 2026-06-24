import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

# Multi-Input Joint Augmentations (V2 Features)

img = torch.rand(3, 256, 256)
# 1. Define bounding boxes using tv_tensors [xmin, ymin, xmax, ymax]
boxes = tv_tensors.BoundingBoxes(
    [[10, 20, 50, 80]], 
    format="XYXY", 
    canvas_size=(256, 256)
)

# 2. Multi-target transformations (RandomResizedCrop)
joint_pipeline = v2.Compose([
    v2.RandomResizedCrop(size=(128, 128), antialias=True),
    v2.RandomVerticalFlip(p=1.0)
])

# 3. Running joint pipeline safely across different targets
trans_img, trans_boxes = joint_pipeline(img, boxes)
print("Transformed Image shape:", trans_img.shape)
print("Transformed Box coordinates:", trans_boxes)