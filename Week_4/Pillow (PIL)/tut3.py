from PIL import Image
import os, sys

im = Image.open("Pil/hopper.ppm")

print(im.format, im.size, im.mode)

#Geometric Transforms

out = im.resize((128, 128))
out = im.rotate(45) # degrees counter-clockwise

out = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
out = im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
out = im.transpose(Image.Transpose.ROTATE_90)
out = im.transpose(Image.Transpose.ROTATE_180)

# Relative Resizing

from PIL import Image, ImageOps
size = (100, 150)
with Image.open("hopper.webp") as im:
    ImageOps.contain(im, size).save("imageops_contain.webp")
    ImageOps.cover(im, size).save("imageops_cover.webp")
    ImageOps.fit(im, size).save("imageops_fit.webp")
    ImageOps.pad(im, size, color="#f00").save("imageops_pad.webp")

    # thumbnail() can also be used,
    # but will modify the image object in place
    im.thumbnail(size)
    im.save("image_thumbnail.webp")
