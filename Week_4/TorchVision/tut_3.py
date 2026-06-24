import torch
from torchvision.transforms import v2

# Color & Intensity Adjustments

img = torch.rand(3, 128, 128)

# 1. ColorJitter, Grayscale, and GaussianBlur
color_pipeline = v2.Compose([
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.Grayscale(num_output_channels=3),
    v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
])

out_img = color_pipeline(img)
print("Output color channels:", out_img.shape[0])