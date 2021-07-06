import numpy as np


class GaussianNoisyData:
    """
    This is an iterator class which adds gaussian noise to the data. It takes as input an iterator and noise level
    """
    def __init__(self, data_iterator, noise_std):
        self._std = noise_std
        self._data_iter = data_iterator
        print(f'[{self.__class__.__name__}] Noise:{noise_std}')

    def __iter__(self):
        return self

    def __next__(self):
        data = next(self._data_iter)
        assert set(data.keys()) == set(['image', 'label'])
        data_np = data['image']._numpy()
        if self._std == 0:
            return {'image': data_np, 'label': data['label']}
        else:
            noise = 1 / 255 * np.random.normal(0, self._std, data_np.shape)
            data_np = noise + data_np
            data_np[data_np > 1] = 1
            data_np[data_np < 0] = 0
            return {'image': data_np, 'label': data['label']}
