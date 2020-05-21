# AnimeGAN

AnimeGAN is my foray into the unchartered GAN territory.

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

## Running

### Training from scratch

[Colab Notebook](https://colab.research.google.com/drive/18FmrptLPQgeSTs_Bg8Fchuz2Ryga0xNe?usp=sharing)

You can train the model using

```
python train.py
```

### Using Pre Trained models

coming soon.

### Generation

To boot up the web interface, run

```
python app.py
```