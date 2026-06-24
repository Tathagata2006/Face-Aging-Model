from PIL import Image

# Image Sequences

with Image.open("animation.gif") as im:
    im.seek(1)  # skip to the second frame

    try:
        while 1:
            im.seek(im.tell() + 1)
            # do something to im
    except EOFError:
        pass  # end of sequence

# List of image filenames
image_filenames = [
    "hopper.jpg",
    "rotated_hopper_270.jpg",
    "rotated_hopper_180.jpg",
    "rotated_hopper_90.jpg",
]

# Open images and create a list
images = [Image.open(filename) for filename in image_filenames]

# Save the images as an animated GIF
images[0].save(
    "animated_hopper.gif",
    append_images=images[1:],
    duration=500,  # duration of each frame in milliseconds
    loop=0,  # loop forever
)

