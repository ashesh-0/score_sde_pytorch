import pickle

import datasets
import numpy as np
import sde_lib
import torch
from base_noisy_data import GaussianNoisyData
from configs.subvp.cifar10_ddpm_continuous import get_config


def get_imgs(t):

    config = get_config()
    train_ds, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    batch = torch.from_numpy(next(iter(train_ds))['image']._numpy()).to(config.device).float()
    batch = batch.permute(0, 3, 1, 2)
    batch = scaler(batch)

    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    t = torch.ones(batch.shape[0], device=batch.device) * t
    mean, std = sde.marginal_prob(batch, t)

    z = torch.randn_like(batch)
    perturbed_data = mean + std[:, None, None, None] * z
    perturbed_data[perturbed_data < -1] = -1
    perturbed_data[perturbed_data > 1] = 1
    return inverse_scaler(perturbed_data).permute(0, 2, 3, 1)


if __name__ == '__main__':
    t = 0.1
    data = get_imgs(t)
    fpath = f'/home/ashesh/ashesh/forward_sde_samples_t-{t}.npy'

    with open(fpath, 'wb') as f:
        pickle.dump(data.cpu().numpy(), f)
    print('')
    print(f'Written to {fpath}')
