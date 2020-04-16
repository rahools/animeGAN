from keras.layers import Conv2D, Conv2DTranspose, Concatenate, BatchNormalization, Input, MaxPooling2D, UpSampling2D, LeakyReLU, Activation, Flatten, Dense
from keras.models import Model

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

# Helper For 1st half
def goingUp(inputLayer, mergeLayer, filters):
    upSample1 = UpSampling2D()(inputLayer)
    upConv1 = Conv2D(filters, 2, padding="same")(upSample1)
    concat1 = Concatenate(axis = 3)([mergeLayer, upConv1])

    conv1 = Conv2D(filters, 2, padding = 'same')(concat1)
    act1 = Activation(LR)(conv1)
    batchNorm1 = BatchNormalization()(act1)

    conv2 = Conv2D(filters, 2, padding = 'same')(batchNorm1)
    act2 = Activation(LR)(conv2)
    batchNorm2 = BatchNormalization()(act2)
    return batchNorm2

# Helper For Output Layer
def goingOut(inputLayer):
    conv1 = Conv2DTranspose(3, 1)(inputLayer)
    output = Activation('tanh')(conv1)
    return output

# Helper For Creating Gen Model
def createGenModel():
    filters = 64
    inputLayer = Input(shape = [64, 64, 3])
    mergeLayer = []
    
    # Total Down 4 | Bottom 1 |Up 4
    
    # Down 1
    mergeLayer1, down1 = goingDown(inputLayer, filters)
    filters *= 2
    mergeLayer.append(mergeLayer1)
    
    # Down 2
    mergeLayer2, down2 = goingDown(down1, filters)
    filters *= 2
    mergeLayer.append(mergeLayer2)
    
    # Down 3
    mergeLayer3, down3 = goingDown(down2, filters)
    filters *= 2
    mergeLayer.append(mergeLayer3)
    
    # Down 4
    mergeLayer4, down4 = goingDown(down3, filters)
    filters *= 2
    mergeLayer.append(mergeLayer4)
    
    # Bottom
    bottom = goingDown(down4, filters, pool = False)    
    
    # Up 1
    filters = filters / 2
    up1 = goingUp(bottom, mergeLayer[-1], filters=int(filters))
    
    # Up 2
    filters = filters / 2
    up2 = goingUp(up1, mergeLayer[-2], filters=int(filters))
    
    # Up 3
    filters = filters / 2
    up3 = goingUp(up2, mergeLayer[-3], filters=int(filters))
    
    # Up 4
    filters = filters / 2
    up4 = goingUp(up3, mergeLayer[-4], filters=int(filters))
    
    # Output
    out = goingOut(up4)
    
    # Create Model
    model = Model(inputLayer, out)
    
    return model

# Helper For Creating Disc Model
def createDiscModel(inputLayer = None):
    filters = 8

    if inputLayer == None:
        inputLayer = Input(shape = [64, 64, 3])

    (_, layer1) = goingDown(inputLayer, filters)

    flat = Flatten()(layer1)

    dense1 = Dense(4)(flat)
    act1 = Activation(LR)(dense1)

    out = Dense(1, activation = 'sigmoid')(act1)

    model = Model(inputLayer, out)

    return model

# Helper to Return all Models Req
def createModels():
    genModel = createGenModel()

    discModel = createDiscModel()

    adrModel = discModel(genModel.output)
    adrModel = Model(genModel.input, adrModel)

    return genModel, discModel, adrModel


