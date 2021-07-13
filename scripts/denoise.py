"""
Here, one can get reconstruction of the images.
We save three set of files:
1. Clean images.
2. Reconstructed images.
3. Noisy images (which were present in the training.)
"""
import gc
import io
import logging
import os

import datasets
import likelihood
import losses
import numpy as np
import sampling
import sde_lib
import tensorflow as tf
import tensorflow_gan as tfgan
import torch
from absl import flags
from configs.subvp.cifar10_ddpm_continuous import get_config
from likelihood import get_likelihood_fn
# Keep the import below for registering all model definitions
from models import ddpm, ncsnpp, ncsnv2
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from run_lib import train
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import restore_checkpoint, save_checkpoint

from scripts.get_marginal_probablity import get_noisy_imgs_from_batch


def write_to_file(fpath, data):
    with tf.io.gfile.GFile(fpath, "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, samples=data)
        fout.write(io_buffer.getvalue())


def denoise(workdir, eval_folder):
    config = get_config()
    eval_dir = os.path.join(workdir, eval_folder)
    tf.io.gfile.makedirs(eval_dir)

    # Build data pipeline
    train_ds, eval_ds, _ = datasets.get_dataset(config,
                                                uniform_dequantization=config.data.uniform_dequantization,
                                                evaluation=True)
    train_iter = iter(eval_ds)
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialize model
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints")

    # Setup SDEs
    assert config.training.sde.lower() == 'subvpsde'
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3

    sampling_shape = (config.eval.batch_size, config.data.num_channels, config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    likelihood_fn = get_likelihood_fn(sde, inverse_scaler)

    begin_ckpt = config.eval.begin_ckpt
    logging.info("begin checkpoint: %d" % (begin_ckpt, ))
    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
        # Wait if the target checkpoint doesn't exist yet
        ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        assert tf.io.gfile.exists(ckpt_filename)

        # Wait for 2 additional mins in case the file exists but is not ready for reading
        ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
        state = restore_checkpoint(ckpt_path, state, device=config.device)
        ema.copy_to(score_model.parameters())

        assert config.eval.enable_sampling
        num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1

        for r in range(num_sampling_rounds):
            logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

            # Get the latent codes
            batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
            batch = batch.permute(0, 3, 1, 2)
            batch = scaler(batch)

            noisy_batch = get_noisy_imgs_from_batch(config.training.start_t, batch, sde, inverse_scaler)
            _, z, _ = likelihood_fn(score_model, noisy_batch, data_t=config.training.start_t)

            this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
            tf.io.gfile.makedirs(this_sample_dir)
            samples, n = sampling_fn(score_model, z=z)
            samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
            samples = samples.reshape((-1, config.data.image_size, config.data.image_size, config.data.num_channels))

            # Write samples to disk or Google Cloud Storage
            write_to_file(os.path.join(this_sample_dir, f"samples_{r}.npz"), samples)
            write_to_file(os.path.join(this_sample_dir, f"clean_data_{r}.npz"),
                          inverse_scaler(batch).permute(0, 2, 3, 1).cpu().numpy())
            write_to_file(os.path.join(this_sample_dir, f"noisy_data_{r}.npz"), noisy_batch.cpu().numpy())
            gc.collect()


if __name__ == '__main__':
    work_dir = '/tmp2/ashesh/ashesh/train_dir'
    eval_dir = '/tmp2/ashesh/ashesh/train_dir/eval_dir'
    denoise(work_dir, eval_dir)
