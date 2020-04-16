from PIL import Image, ImageDraw
import random

for trueImg in range(1, 21552):
    if trueImg % 1000 == 0:
        print(f'{trueImg * 100 / 21551}% done.')
    for maskNum in range(1, 2):
        maskImg = Image.open(f'data/input/true/{trueImg}.png')
        rec = ImageDraw.Draw(maskImg)
        x0 = random.randrange(8, 40)
        y0 = random.randrange(8, 32)
        rec.rectangle([(x0, y0), (x0 + 16, y0 + 16)], fill = 'black')
        del rec
        maskImg.save(f'data/input/mask/{trueImg}.png')