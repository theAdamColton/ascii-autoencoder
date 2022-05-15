# Code from https://github.com/8bitavenue/Python-Image-Character-Generator/blob/master/generate.py

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import os
from random import randint


def generateCharacters(
    fontpath: str, output_dir: str, characters, background_color, font_size, image_size 
):
    os.makedirs(output_dir, exist_ok=True)
    for i, character in enumerate(characters):
        # Create character image :
        char_image = Image.new("L", (image_size, image_size), background_color)

        # Draw character image
        draw = ImageDraw.Draw(char_image)

        # Specify font : Resource file, font size
        font = ImageFont.truetype(fontpath, font_size)

        # Get character width and height
        (font_width, font_height) = font.getsize(character)

        # Calculate x position
        x = (image_size - font_width) / 2

        # Calculate y position
        y = (image_size - font_height) / 2

        # Draw text : Position, String,
        # Options = Fill color, Font
        draw.text(
            (x, y), character, (245 - background_color) + randint(0, 10), font=font
        )

        # Final file name
        file_name = os.path.join(output_dir, "{}.png".format(i))
        # Save image
        char_image.save(file_name)

        # Print character file name
        print(file_name)


# Directory containing fonts
font_path = "/usr/local/share/fonts/Bitstream Vera Sans Mono Bold Nerd Font Complete Mono.ttf"

# Output
output_dir = "out/"

ascii_codes = range(32, 127)
characters = list(map(chr, ascii_codes))

print(characters)

white_color = 255
black_color = 0

font_size = 40
image_size = 64

if __name__ in {"__main__", "__console__"}:
    generateCharacters(
        font_path, output_dir, characters, white_color, font_size, image_size, 
    )
