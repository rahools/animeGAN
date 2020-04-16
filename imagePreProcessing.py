# Script that returns imgs to training script.
# Script works by randomly selecting, resizing, and converting a random list of imgs to a np array. 
import os
import numpy as np
from PIL import Image


# Img list filter func
def imgFilter(img):
    if img.endswith('.png'):
        return True
    return False

# return img data
def dataBatch(dataLoc = 'data/input', imgType = 'true', batchSize = 5, dims = [64, 64]):
    # check if user require true img or mask img
    if imgType == 'true':
        # Load Train Img
        trainImages = os.listdir('data/input/true')
        volumeImages = list(filter(imgFilter, trainImages))
    elif imgType == 'mask':
        # Load Mask Img
        maskImages = os.listdir('data/input/mask')
        volumeImages = list(filter(imgFilter, maskImages))

    # create a np array of 'batchSize' imgs
    for _ in range(batchSize):
        # randomly select imgs
        imgList = list(np.random.choice(volumeImages, batchSize))
        for imgName in imgList:
            # load, resize, normalize, and covert img to np array
            originalImage = Image.open(f'{dataLoc}/{imgType}/{imgName}')    # load
            resizedImage = np.array(originalImage.resize(dims)) # resize
            arrayImage = resizedImage / 255 # normalize
            img = arrayImage[np.newaxis, :, :, :]

            if imgName == imgList[0]:
                data = img
            else:
                data = np.append(data, img, axis = 0)

    return data

# return img data and its mask
def dataBatchGen(dataLoc = 'data/input', batchSize = 5, dims = [64, 64]):
    # load img list
    trainImages = os.listdir('data/input/true')
    volumeImages = list(filter(imgFilter, trainImages))

    # create a np array of 'batchSize' imgs
    for _ in range(batchSize):
        imgList = list(np.random.choice(volumeImages, batchSize))
        for imgName in imgList:
            # true
            # load, resize, normalize, and covert img to np array
            originalImage = Image.open(f'{dataLoc}/true/{imgName}')
            resizedImage = np.array(originalImage.resize(dims))
            arrayImage = resizedImage / 255
            img = arrayImage[np.newaxis, :, :, :]

            if imgName == imgList[0]:
                data1 = img
            else:
                data1 = np.append(data1, img, axis = 0)

            # mask
            # load, resize, normalize, and covert img to np array
            originalImage = Image.open(f'{dataLoc}/mask/{imgName}')
            resizedImage = np.array(originalImage.resize(dims))
            arrayImage = resizedImage / 255
            img = arrayImage[np.newaxis, :, :, :]

            if imgName == imgList[0]:
                data2 = img
            else:
                data2 = np.append(data2, img, axis = 0)

    # return as true, mask
    return data1, data2

if __name__ == '__main__':
    print(dataBatch().shape)