from keras.layers import Conv2D, Conv2DTranspose, Concatenate, BatchNormalization, Input, MaxPooling2D, UpSampling2D, LeakyReLU, Activation, Flatten, Dense, Reshape
from keras.models import Model

# Global LR object
LR = LeakyReLU(alpha=0.1)
LR.__name__ = 'leakyReLU'

# Model Creater Helper Func
# Helper For 1st half
def goingDown(inputLayer, filters, pool=True):
    conv1 = Conv2D(filters, 2, padding = 'same')(inputLayer)
    act1 = Activation(LR)(conv1)
    batchNorm1 = BatchNormalization()(act1)

    conv2 = Conv2D(filters, 2, padding = 'same')(batchNorm1)
    act2 = Activation(LR)(conv2)
    batchNorm2 = BatchNormalization()(act2)

    if pool:
        pool1 = MaxPooling2D(2)(batchNorm2)
        return batchNorm2, pool1
    return batchNorm2

# Helper For Creating Gen Model
def createGenModel():
    filters = 64
    inputLayer = Input(shape = [64, 64, 3])

    convT1 = Conv2DTranspose(filters, 2, padding = 'same')(inputLayer)
    act1 = Activation(LR)(convT1)
    batchNorm1 = BatchNormalization()(act1)

    convT2 = Conv2DTranspose(filters * 2, 2, padding = 'same')(batchNorm1)
    act2 = Activation(LR)(convT2)
    batchNorm2 = BatchNormalization()(act2)

    convT3 = Conv2DTranspose(3, 2, padding = 'same')(batchNorm2)
    out = Activation('tanh')(convT3)
    
    # Create Model
    model = Model(inputLayer, out)
    
    return model

# Helper For Creating Disc Model
def createDiscModel(inputLayer = None):
    filters = 8

    if inputLayer == None:
        inputLayer = Input(shape = [64, 64, 3])

    (_, layer1) = goingDown(inputLayer, filters)
    (_, layer2) = goingDown(layer1, filters)

    flat = Flatten()(layer2)

    dense1 = Dense(64)(flat)
    act1 = Activation(LR)(dense1)

    dense2 = Dense(64)(act1)
    act2 = Activation(LR)(dense2)

    out = Dense(1, activation = 'sigmoid')(act2)

    model = Model(inputLayer, out)

    return model

# Helper to Return all Models Req
def createModels():
    genModel = createGenModel()

    discModel = createDiscModel()

    adrModel = discModel(genModel.output)
    adrModel = Model(genModel.input, adrModel)

    return genModel, discModel, adrModel

if __name__ == '__main__':
    m = createGenModel()
    m.summary()


