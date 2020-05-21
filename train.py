import os
import torch
import numpy as np

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import Generator, Discriminator

import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

from time import time
import pickle


# Use GPU is available
if torch.cuda.is_available():
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')


def trainHelper(WORK_DIR, genModel, discModel, epochs, verbose):
    WORK_DIR = WORK_DIR + '/'
    INPUT_LOC = WORK_DIR + '/input/'

    # Create Image dataset
    dataset = dset.ImageFolder(root = INPUT_LOC + 'true/', transform = transforms.Compose([
                                transforms.Resize(64),
                                transforms.CenterCrop(64),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))

    # Setup a Data-loader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle = True, num_workers = 2)


    # Initialize loss function
    lossFunc = nn.BCELoss()

    # Setup Adam optimizers for both G and D
    lr = .0001
    beta1 = .5
    discOptimizer = optim.Adam(discModel.parameters(), lr = lr, betas=(beta1, 0.999))
    genOptimizer = optim.Adam(genModel.parameters(), lr = lr, betas=(beta1, 0.999))


    # variable to store iters
    iterNo = 0


    for epoch in range(epochs):
        for idx, i in enumerate(dataloader):
            # Disc Model
            discModel.zero_grad()

            ## loading true imgs
            trueImg = i[0].to(device)
            imgCount = trueImg.size(0)
            label = torch.ones((imgCount,), device = device)

            ## training on true imgs
            output = discModel(trueImg).view(-1)
            discErrorReal = lossFunc(output, label)
            discErrorReal.backward()
            dx = output.mean().item()

            ## generating fake imgs
            noise = torch.randn(imgCount, 100, 1, 1, device=device)
            fakeImgs = genModel(noise)
            label = torch.zeros((imgCount,), device = device)

            ## training on gen fake imgs
            output = discModel(fakeImgs.detach()).view(-1)
            discErrorFake = lossFunc(output, label)
            discErrorFake.backward()
            dgx_d = output.mean().item()

            ## optimize grads
            discOptimizer.step()

            # Gen Model
            genModel.zero_grad()

            ## generating labels
            label = torch.ones((imgCount,), device = device)

            ## taining on previously generated fake images, but without 'detach()'
            output = discModel(fakeImgs).view(-1)
            genError = lossFunc(output, label)
            genError.backward()
            dgx_g = output.mean().item()

            ## optimize grads
            genOptimizer.step()


            # Increase iter
            iterNo += 1

            if iterNo % 100 == 0 and verbose:
                # print details | every 100 iters
                print(f'[{epoch} / {epochs}][{idx} / {len(dataloader)}]')
                print(f'Disc Loss: {(discErrorFake + discErrorReal).item():.5f}  Gen Loss: {genError.item():.5f}')
                print(f'True Output: {dx:.5f}  Fake Output in Disc: {dgx_d:.5f}  Fake Output in Gen: {dgx_g:.5f}')
                print('')

            if iterNo % 1000 == 0:
                # save checkpoint | every 1000 iters
                print('Checkpointing the model!')
                torch.save({
                    'epoch' : epoch,
                    'discStateDict' : discModel.state_dict(),
                    'genStateDict' : genModel.state_dict(),
                    'discOptimizerStateDict' : discOptimizer.state_dict(),
                    'genOptimizerStateDict' : genOptimizer.state_dict(),
                    'discLoss' : (discErrorFake + discErrorReal).item(),
                    'genLoss' : genError.item()
                }, f'{WORK_DIR}checkpoint/{int(time())}-checkpoint.tar')
                print('Done!!!')
                print('')

    # save final models after taining ends
    torch.save(discModel.state_dict(), f'{WORK_DIR}savedModel/{int(time())}-disc.pt')
    torch.save(genModel.state_dict(), f'{WORK_DIR}savedModel/{int(time())}-gen.pt')

def train():
    # Location to data location
    WORK_DIR = '/animeGAN/'

    # load models
    discModel = Discriminator().to(device)
    genModel = Generator().to(device)

    epochs = 10
    verbose = False

    trainHelper(WORK_DIR, genModel, discModel, epochs, verbose)

if __name__ == '__main__':
    train()


