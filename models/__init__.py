import pickle

from .salience_network import SalienceNetwork
from .harmonicCNN import HarmonicCNN
from .unet import UNet


__all__ = [
    "SalienceNetwork",
    "HarmonicCNN",
    "UNet"
]


def load_from_dir(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model