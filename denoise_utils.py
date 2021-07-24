import io
import os

import numpy as np
import tensorflow as tf
import torch

import datasets
import losses
import sampling
import sde_lib
from likelihood import get_likelihood_fn
# Keep the import below for registering all model definitions
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from utils import restore_checkpoint


def dump_data(batch, noisy_batch, samples, output_dir, sampling_round):
    samples = (samples.cpu().numpy()).astype(np.uint8)
    batch = (batch.cpu().numpy()).astype(np.uint8)
    noisy_batch = (noisy_batch.cpu().numpy()).astype(np.uint8)

    # Write samples to disk or Google Cloud Storage
    write_to_file(os.path.join(output_dir, f"samples_{sampling_round}.npz"), samples)
    write_to_file(os.path.join(output_dir, f"clean_data_{sampling_round}.npz"), batch)
    write_to_file(os.path.join(output_dir, f"noisy_data_{sampling_round}.npz"), noisy_batch)


def convert_to_img(data, inverse_scaler):
    data[data < -1] = -1
    data[data > 1] = 1
    return inverse_scaler(data).permute(0, 2, 3, 1)


def post_process_data(setup, batch, noisy_batch, samples):
    samples = samples.permute(0, 2, 3, 1)
    samples = torch.clamp(samples, min=0, max=1)
    samples = 255 * samples

    noisy_batch = convert_to_img(noisy_batch, setup.inverse_scaler)
    noisy_batch = 255 * noisy_batch

    batch = setup.inverse_scaler(batch).permute(0, 2, 3, 1)
    batch = 255 * batch
    return batch, noisy_batch, samples


class Setup:
    """
    Holds all the variables which are needed to do anything, it includes SDE, scaler, inverse scaler etc.
    """
    def __init__(self, config, workdir):
        self.config = config
        self.workdir = workdir
        # Build data pipeline
        train_ds, eval_ds, _ = datasets.get_dataset(config,
                                                    uniform_dequantization=config.data.uniform_dequantization,
                                                    evaluation=True)
        self.train_ds = train_ds
        self.eval_ds = eval_ds
        self.start_t = config.training.start_t
        # Create data normalizer and its inverse
        self.scaler = datasets.get_data_scaler(config)
        self.inverse_scaler = datasets.get_data_inverse_scaler(config)

        # Initialize model
        self.score_model = mutils.create_model(config)
        self.optimizer = losses.get_optimizer(config, self.score_model.parameters())
        self.ema = ExponentialMovingAverage(self.score_model.parameters(), decay=config.model.ema_rate)
        self.state = dict(optimizer=self.optimizer, model=self.score_model, ema=self.ema, step=0)

        if self.start_t is None:
            self.start_t = 1e-3
        self.existing_noise_t = config.data.existing_noise_t
        if self.existing_noise_t is None:
            self.existing_noise_t = 0
        else:
            assert self.existing_noise_t <= self.start_t, (f'existing_noise_t:{self.existing_noise_t} > '
                                                           f'start_t:{self.start_t}')

        # Setup SDEs
        assert config.training.sde.lower() == 'subvpsde'
        self.sde = sde_lib.subVPSDE(beta_min=config.model.beta_min,
                                    beta_max=config.model.beta_max,
                                    N=config.model.num_scales,
                                    start_t=self.start_t,
                                    existing_noise_t=self.existing_noise_t)
        sampling_eps = 1e-3

        self.sampling_shape = (config.eval.batch_size, config.data.num_channels, config.data.image_size,
                               config.data.image_size)
        self.sampling_fn = sampling.get_sampling_fn(config, self.sde, self.sampling_shape, self.inverse_scaler,
                                                    sampling_eps)

        self.likelihood_fn = get_likelihood_fn(self.sde, self.inverse_scaler)

    def load_weights(self, checkpoint_idx):
        checkpoint_dir = os.path.join(self.workdir, "checkpoints")

        # Wait if the target checkpoint doesn't exist yet
        ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(checkpoint_idx))
        assert tf.io.gfile.exists(ckpt_filename)

        # Wait for 2 additional mins in case the file exists but is not ready for reading
        ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{checkpoint_idx}.pth')
        self.state = restore_checkpoint(ckpt_path, self.state, device=self.config.device)
        self.ema.copy_to(self.score_model.parameters())


def write_to_file(fpath, data):
    with tf.io.gfile.GFile(fpath, "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, samples=data)
        fout.write(io_buffer.getvalue())
