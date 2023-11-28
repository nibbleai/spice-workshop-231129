import numpy as np


def cyclical(data, *, cycle_length):
    ratio = data * (2 * np.pi / cycle_length)
    return {
        'sin': np.sin(ratio),
        'cos': np.cos(ratio)
    }
