import os
import numpy as np
from PIL import Image

from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split


# Data Generator for Training and Validation data
volumeImages = range(1, 21551)
trainImages, validationImages = train_test_split(volumeImages, train_size=0.8, test_size=0.2)

def dataGenerator(dataLoc = 'data/input', batchSize = 8, dims = [64, 64], val = False):
    if val:
        imageSet = volumeImages
    else:
        imageSet = volumeImages

    while True:
        ix = np.random.choice(np.arange(len(imageSet)), batchSize)
        imgs = []
        labels = []
        for i in ix:
            # images
            originalImage = load_img(f'{dataLoc}/true/{imageSet[i]}.png')
            resizedImage = np.array(Image.fromarray(originalImage).resize(dims + [3]))
            # resizedImage = imresize(originalImage, dims + [3])
            arrayImage = img_to_array(resizedImage) / 255
            imgs.append(arrayImage)
            
            # masks
            originalMask = load_img(f'{dataLoc}/mask/{imageSet[i]}_mask1.png')
            resizedMask = np.array(Image.fromarray(originalMask).resize(dims + [3]))
            # resizedMask = imresize(originalMask, dims + [3])
            arrayMask = img_to_array(resizedMask) / 255
            labels.append(arrayMask)
            
        imgs = np.array(imgs)
        labels = np.array(labels)
        yield imgs, labels


def dataBatch(dataLoc = 'data/input', imgType = 'train', batchSize = 5, dims = [64, 64]):
    if imgType == 'train':
        imgType = 'true'
    else:
        imgType = 'mask'

    for _ in range(batchSize):
        imgList = np.random.randint(1, 21000, batchSize)
        for imgNumber in imgList:
            originalImage = Image.open(f'{dataLoc}/{imgType}/{volumeImages[imgNumber]}.png')
            resizedImage = np.array(originalImage.resize(dims))
            arrayImage = resizedImage / 255
            # print(arrayImage.shape)
            img = arrayImage[np.newaxis, :, :, :]
            # print(img.shape)

            if imgNumber == imgList[0]:
                data = img
            else:
                data = np.append(data, img, axis = 0)

    return data

def dataBatchGen(dataLoc = 'data/input', batchSize = 5, dims = [64, 64]):
    for _ in range(batchSize):
        imgList = np.random.randint(1, 21000, batchSize)
        for imgNumber in imgList:
            # true
            originalImage = Image.open(f'{dataLoc}/true/{volumeImages[imgNumber]}.png')
            resizedImage = np.array(originalImage.resize(dims))
            arrayImage = resizedImage / 255
            img = arrayImage[np.newaxis, :, :, :]

            if imgNumber == imgList[0]:
                data1 = img
            else:
                data1 = np.append(data1, img, axis = 0)

            # mask
            originalImage = Image.open(f'{dataLoc}/mask/{volumeImages[imgNumber]}.png')
            resizedImage = np.array(originalImage.resize(dims))
            arrayImage = resizedImage / 255
            img = arrayImage[np.newaxis, :, :, :]

            if imgNumber == imgList[0]:
                data2 = img
            else:
                data2 = np.append(data2, img, axis = 0)

    return data1, data2

if __name__ == '__main__':
    print(dataBatch().shape)