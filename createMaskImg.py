# Script that creates trainging masks.
# Script works by drawing a black rec over true img. 
from PIL import Image, ImageDraw
import random
import os
import progressbar

def main():
    # Get a list of items in the dir
    imgList = os.listdir('data/input/true') 

    for i in progressbar.progressbar(imgList):
        # Check if the file is image or not
        if i.endswith('.png'):
            # Open the true img
            maskImg = Image.open(f'data/input/true/{i}')

            # Draw a black rectangle over true img 
            rec = ImageDraw.Draw(maskImg)
            x0 = random.randrange(8, 40)    # random sizing & position
            y0 = random.randrange(8, 32)    # random sizing & position
            rec.rectangle([(x0, y0), (x0 + 16, y0 + 16)], fill = 'black')
            del rec

            # Save mask
            maskImg.save(f'data/input/mask/{i}')


if __name__ == '__main__':
    main()