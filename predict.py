import os
import numpy as np
from PIL import Image

from keras.models import load_model
from keras.layers import LeakyReLU

def predict():
    LR = LeakyReLU(alpha=0.1)
    LR.__name__ = 'leakyReLU'

    # Load Model
    model = load_model('data/modelSaved/genModel.h5', custom_objects = {"leakyReLU": LR})

    # Create Predict Data & predict & save
    predImages = os.listdir('data/topredict')

    for img in predImages:
        originalImage = Image.open(f'data/topredict/{img}')
        resizedImage = np.array(originalImage.resize([64, 64]))
        arrayImage = resizedImage / 255
        img2p = arrayImage[np.newaxis, :, :, :]

        predArr = model.predict(img2p)

        predArr = predArr.swapaxes(1, 2)
        predImg = Image.fromarray(predArr[0, :, :, :], 'RGB')
        predImg.save(f'data/predicted/{img}')

if __name__ == '__main__':
    predict()