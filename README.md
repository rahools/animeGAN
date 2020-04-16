# AnimeGAN

AnimeGAN is my foray into the unchartered GAN territory. My goal for this project is to be able to construct partially erased anime face images. 

NOTE: THE PROJECT IS STILL UNDER DEVELOPMENT.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

python 3.7

### Installing

First, you would have to install virtualenv.

```
pip install virtualenv
```

Create and activate a python virtualenv.

```
python -m venv animegan
.\animegan\Scripts\activate
```

Install the required python packages.

```
pip install -r requirements.txt
```

### Generating Training Data

Download and populate the data directory, I used this [dataset](https://github.com/Mckinsey666/Anime-Face-Dataset).
```
data/input/true
```

Generate training masks using
```
python createMaskImg.py
```

## Running

### Training from scratch

You can train the model using

```
python trainModel.py
```

### Using Pre Trained models

comming soon.

### Predicting

Paste an image that has to the construct in 

```
data/topredict
```

To construct the image, run

```
python predict.py
```

you can now see the results in 

```
data/predicted
```