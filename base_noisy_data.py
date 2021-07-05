import numpy as np


class GaussianNoisyData:
    """
    This is an iterator class which adds gaussian noise to the data. It takes as input an iterator and noise level
    """
    def __init__(self, data_iterator, noise_std):
        self._std = noise_std
        self._data_iter = data_iterator

    def __iter__(self):
        return self

    def __next__(self):
        data = next(self._data_iter)
        noise = np.random.normal(0, self._std, data.shape)
        return noise + data
