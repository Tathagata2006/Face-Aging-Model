from PIL import Image
import os, sys

im = Image.open("Pil/hopper.ppm")

print(im.format, im.size, im.mode)

# Color Transforms

with Image.open("hopper.ppm") as im:
    im = im.convert("L")
    
# Applying Filters

from PIL import ImageFilter
out = im.filter(ImageFilter.DETAIL)

# multiply each pixel by 20
out = im.point(lambda i: i * 20) 

# split the image into individual bands
source = im.split()

R, G, B = 0, 1, 2

# select regions where red is less than 100
mask = source[R].point(lambda i: i < 100 and 255)

# process the green band
out = source[G].point(lambda i: i * 0.7)

# paste the processed band back, but only where red was < 100
source[G].paste(out, None, mask)

# build a new multiband image
im = Image.merge(im.mode, source)