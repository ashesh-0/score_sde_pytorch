import pickle

import datasets
import numpy as np
from base_noisy_data import GaussianNoisyData
from configs.subvp.cifar10_ddpm_continuous import get_config


def get_imgs(std_val):

    config = get_config()
    train_ds, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)
    train_ds = GaussianNoisyData(iter(train_ds), std_val)
    return next(train_ds)


if __name__ == '__main__':
    std_val = 25
    data = get_imgs(std_val)
    fpath = f'/home/ashesh/ashesh/noisy_samples_std-{std_val}.npy'
    with open(fpath, 'wb') as f:
        pickle.dump(data['image'], f)
    print('')
    print(f'Written to {fpath}')
