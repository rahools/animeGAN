from model import Generator
from flask import Flask, render_template

import torch
import torchvision.utils as vutils

import os
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
from time import time

# Flask Object
app = Flask(__name__)

# Setup torch device
## Use GPU if available
if torch.cuda.is_available():
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')

# Getting our generator loaded up
## Get the latest Saved Model
modelList = os.listdir('./savedModel')
modelList.sort()
## Load the saved model
generator = Generator()
generator.load_state_dict(torch.load(f'./savedModel/{modelList[-1]}', map_location='cpu'))
## push the model to our torch device
generator = generator.to(device)
## put model in eval() mode
generator = generator.eval()


# Flask routes
@app.route("/")
def home():
    return render_template('index.html')

@app.route("/generate")
def generate():
    # Generating anime character from random noise
    noise = torch.randn(64, 100, 1, 1, device = device)
    generatedImgArray = generator(noise).detach().cpu()

    # plotting img
    f, ax = plt.subplots(figsize=(11, 9))
    ax.axis("off")
    ax.imshow(np.transpose(vutils.make_grid(generatedImgArray, padding=2, normalize=True), (1,2,0)))

    # saving plot image
    img = BytesIO()
    f.savefig(img, format='png')
    plt.close(f)

    img.seek(0)
    b64img = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('generatorOut.html', genImage = b64img)

if __name__ == '__main__':
    app.run(debug=True)
