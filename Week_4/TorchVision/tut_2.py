import torch
from torchvision.transforms import v2

# Geometric Transformations

img = torch.rand(3, 256, 256)

# 1. Resize, CenterCrop, and RandomHorizontalFlip
geom_pipeline = v2.Compose([
    v2.Resize(size=(128, 128), antialias=True),
    v2.CenterCrop(size=(100, 100)),
    v2.RandomHorizontalFlip(p=0.5)
])

out_img = geom_pipeline(img)
print("Output geometric shape:", out_img.shape)