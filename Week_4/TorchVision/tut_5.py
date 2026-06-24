import torch
from torchvision.transforms import v2

# Advanced Compositions & Normalization

img = torch.rand(3, 224, 224)

# 1. RandomApply, RandomChoice, and Normalize
final_pipeline = v2.Compose([
    v2.RandomApply([v2.ColorJitter(brightness=0.5)], p=0.3),
    v2.RandomChoice([v2.RandomRotation(degrees=45), v2.RandomAffine(degrees=0, translate=(0.1, 0.1))]),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

out_img = final_pipeline(img)
print("Final normalized tensor mean statistics:", out_img.mean(dim=[-2, -1]))