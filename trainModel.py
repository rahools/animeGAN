from model import createModels
from imagePreProcessing import dataBatch, dataBatchGen
from PIL import Image

from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model
from keras.layers import LeakyReLU

import time
import numpy as np
import os
import matplotlib.pyplot as plt

# Img list filter func
def imgFilter(img):
    if img.endswith('.png'):
        return True
    return False

def trainAnimeGen(batchSize = 5, epoch = 10, saveInterval = 2, modelSave = True, useSavedModel = False, verbose = False, picsave = False):
    # Model creater helper call
    if useSavedModel:
        LR = LeakyReLU(alpha=0.1)
        LR.__name__ = 'leakyReLU'

        genModel = load_model('data/modelSaved/genModel.h5', custom_objects = {"leakyReLU": LR})
        discModel = load_model('data/modelSaved/discModel.h5', custom_objects = {"leakyReLU": LR})
        adrModel = load_model('data/modelSaved/adrModel.h5', custom_objects = {"leakyReLU": LR})
    else:
        genModel, discModel, adrModel = createModels()

    # Create Disc Model
    optimizer = RMSprop(lr = 0.0002, decay = 6e-8)
    discModel.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

    # Create Gen Model
    optimizer = RMSprop(lr=0.0001, decay=3e-8)
    genModel.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

    # Create Ad Model
    optimizer = RMSprop(lr=0.0001, decay=3e-8)
    adrModel.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

    for i in range(epoch):
        # Load data
        xTrue = dataBatch(imgType = 'true', batchSize = batchSize)
        xMask = dataBatch(imgType = 'mask', batchSize = batchSize)

        # Generate anime faces from masks
        xGen = genModel.predict(xMask)

        # Train Discriminator
        x = np.append(xTrue, xGen, axis = 0)
        y = np.ones([2 * batchSize, 1])
        y[batchSize:, :] = 0
        dLoss = discModel.train_on_batch(x, y)

        # Train Adr
        xMask = dataBatch(imgType = 'mask', batchSize = batchSize)
        y = np.ones([batchSize, 1])
        aLoss = adrModel.train_on_batch(xMask, y)

        # Train Gen
        y, x = dataBatchGen()
        gLoss = genModel.train_on_batch(x, y)

        # Logging losses
        if verbose:
            print(f'{i + 1} D Loss: {dLoss[0]}  Acc: {dLoss[1]}')
            print(f'{i + 1} A Loss: {aLoss[0]}  Acc: {aLoss[1]}')
            print(f'{i + 1} G Loss: {gLoss[0]}  Acc: {gLoss[1]}')

        if (saveInterval > 0) and ((i + 1) % saveInterval == 0):
            print(f'Save Interval{i + 1} Time')
            iNum = i + 1
            predImages = os.listdir('data/topredict')
            predImages = list(filter(imgFilter, predImages))
            for i in predImages:
                # Open test img
                originalImage = Image.open(f'data/topredict/{i}')
                resizedImage = np.array(originalImage.resize([64, 64]))
                arrayImage = resizedImage / 255
                img = arrayImage[np.newaxis, :, :, :]

                # Pred img
                predArr = genModel.predict(img)

                # Save img
                imgName = i.split('.')[0]
                try:
                    if picsave:
                        # predArr = predArr.swapaxes(1, 2)
                        predImg = Image.fromarray(predArr[0, :, :, :], 'RGB')
                        predImg.save(f'data/predicted/{imgName}Interval{iNum}.png')
                except:
                    pass

            # save Model
            if modelSave:
                genModel.save('data/modelSaved/genModel.h5')
                discModel.save('data/modelSaved/discModel.h5')
                adrModel.save('data/modelSaved/adrModel.h5')

def summary(useSavedModel = False):
    if useSavedModel:
        LR = LeakyReLU(alpha=0.1)
        LR.__name__ = 'leakyReLU'

        genModel = load_model('data/modelSaved/genModel.h5', custom_objects = {"leakyReLU": LR})
        discModel = load_model('data/modelSaved/discModel.h5', custom_objects = {"leakyReLU": LR})
        adrModel = load_model('data/modelSaved/adrModel.h5', custom_objects = {"leakyReLU": LR})
    else:
        genModel, discModel, adrModel = createModels()

    discModel.summary()
    genModel.summary()
    adrModel.summary()

    
if __name__ == '__main__':
    # summary(useSavedModel = False)
    trainAnimeGen(batchSize = 1, epoch = 10000, saveInterval = 100, useSavedModel = True, verbose = True, picsave = True)