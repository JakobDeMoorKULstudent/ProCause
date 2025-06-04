import numpy as np
import pickle

def save_data(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_data(path):
    with open(path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data