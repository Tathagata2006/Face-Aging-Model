import torch
from torchvision.transforms import v2

# Core Setup & Conversions

# 1. Dummy 8-bit image data (Channels, Height, Width)
raw_img = torch.randint(0, 256, size=(3, 128, 128), dtype=torch.uint8)

# 2. Conversion & Type Casting (ToImage, ToDtype)
setup_pipeline = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

processed_img = setup_pipeline(raw_img)
print("Processed image shape:", processed_img.shape)
print("Data type:", processed_img.dtype)
print("Value range max:", processed_img.max().item())