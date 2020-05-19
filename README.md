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

Download and populate the data directory, I used this [dataset](https://www.kaggle.com/splcher/animefacedataset).
```
input/true/1
```

## Milestone 1: Face Generation from Noise | Completed

To ease up the development, I decided to break the project into bite-size pieces. So for the first milestone, I choose to set up a GAN model such that it would automatically generate faces from random noise. [link](https://github.com/rahools/animeGAN/tree/milestone1)

## Milestone 2: Converting Input Image

Next in line, I would be tackling handling partially erased input images into a latent space embedding, which further could be feed into the GAN model.


