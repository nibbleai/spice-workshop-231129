import pickle
from sklearn.base import BaseEstimator
from spice import Generator
from spice import serializer

from src.directories import directories

MODEL_FILENAME = 'model.pkl'
GENERATOR_FILENAME = 'generator.spice'

MODEL_PATH = directories.artifacts_dir / MODEL_FILENAME
GENERATOR_PATH = directories.artifacts_dir / GENERATOR_FILENAME


def save_model(model: BaseEstimator):
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)


def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def save_generator(generator: Generator):
    with open(GENERATOR_PATH, 'wb') as f:
        serializer.dump(generator, f)


def load_generator():
    with open(GENERATOR_PATH, 'rb') as f:
        return serializer.load(f)
