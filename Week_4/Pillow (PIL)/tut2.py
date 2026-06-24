from PIL import Image
import os, sys

im = Image.open("Pil/hopper.ppm")

print(im.format, im.size, im.mode)

# Cutting, pasting, and merging images

box = (0, 0, 64, 64)
region = im.crop(box)

region = region.transpose(Image.Transpose.ROTATE_180)
im.paste(region, box)

# Rolling an image

def roll(im: Image.Image, delta: int) -> Image.Image:
    """Roll an image sideways."""
    xsize, ysize = im.size

    delta = delta % xsize
    if delta == 0:
        return im

    part1 = im.crop((0, 0, delta, ysize))
    part2 = im.crop((delta, 0, xsize, ysize))
    im.paste(part1, (xsize - delta, 0, xsize, ysize))
    im.paste(part2, (0, 0, xsize - delta, ysize))

    return im

# Merging Image

def merge(im1: Image.Image, im2: Image.Image) -> Image.Image:
    w = im1.size[0] + im2.size[0]
    h = max(im1.size[1], im2.size[1])
    im = Image.new("RGBA", (w, h))

    im.paste(im1)
    im.paste(im2, (im1.size[0], 0))

    return im

