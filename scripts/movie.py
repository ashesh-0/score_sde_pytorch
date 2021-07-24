"""
Create a movie
"""
import gc
import logging
import os

import numpy as np
import tensorflow as tf
import torch
from absl import app, flags
from denoise_utils import Setup, dump_data, post_process_data, write_to_file
from eval_metrics import EvalMetrics
from ml_collections.config_flags import config_flags
from models import ddpm
from torchvision.io import write_video
from torchvision.utils import make_grid
# Keep the import below for registering all model definitions
from tqdm import tqdm

from scripts.get_marginal_probablity import get_noisy_imgs_from_batch

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")

flags.DEFINE_float('latent_start_t', 1e-3, 'Latent start_t')
flags.DEFINE_float('sampling_end_t', 1e-3, 'Sampling end_t')
flags.DEFINE_bool('skip_forward_integration', False, 'If True, the the forward integration is not used.'
                  ' We just use the reverse integration')

# flags.DEFINE_string("eval_folder", "eval", "The folder name for storing evaluation results")
flags.mark_flags_as_required(["workdir", "config"])

FLAGS = flags.FLAGS


def main(argv):
    workdir = FLAGS.workdir
    latent_start_t = FLAGS.latent_start_t
    sampling_end_t = FLAGS.sampling_end_t
    config = FLAGS.config
    start_t = config.training.start_t
    skip_forward_integration = FLAGS.skip_forward_integration

    eval_dir = (f'{workdir}/video/eval_dir_nt_{start_t}_'
                f'lt_{latent_start_t}_st_{sampling_end_t}_skipF_{int(skip_forward_integration)}')
    print('')
    print('Evaluation will be saved to ', eval_dir)
    print('')

    setup = Setup(config, workdir)
    tf.io.gfile.makedirs(eval_dir)

    train_iter = iter(setup.eval_ds)
    # checkpoint_dir = os.path.join(workdir, "checkpoints")
    begin_ckpt = config.eval.begin_ckpt
    logging.info("begin checkpoint: %d" % (begin_ckpt, ))
    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
        setup.load_weights(ckpt)

        assert config.eval.enable_sampling

        this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
        tf.io.gfile.makedirs(this_sample_dir)
        logging.info("sampling -- ckpt: %d" % (ckpt))

        batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
        batch = batch.permute(0, 3, 1, 2)
        batch = setup.scaler(batch)

        noisy_batch = get_noisy_imgs_from_batch(start_t - setup.existing_noise_t, batch, setup.sde,
                                                setup.inverse_scaler)
        assert skip_forward_integration == True

        step = -0.01
        intervals = np.arange(start=latent_start_t, stop=sampling_end_t, step=step)
        if intervals[-1] - sampling_end_t > 0.01:
            intervals = np.append(intervals, sampling_end_t)

        frames = []
        pbar = tqdm(intervals[1:])
        for temp_sampling_end_t in pbar:
            pbar.set_description(f"start:{latent_start_t:.3f} end:{temp_sampling_end_t:.3f}")
            temp_z, _ = setup.sampling_fn(
                setup.score_model,
                z=noisy_batch,
                end_t=temp_sampling_end_t,
                start_t=latent_start_t,
            )
            frames.append(temp_z.cpu())

        samples = torch.cat(frames, dim=0)
        # Convert them to [0,255] and uniform shape
        batch, noisy_batch, samples = post_process_data(setup, batch, noisy_batch, samples)
        batch = batch.permute(0, 3, 1, 2)
        noisy_batch = noisy_batch.permute(0, 3, 1, 2)
        samples = samples.permute(0, 3, 1, 2)

        bsize = frames[0].shape[0]
        nrow = int(np.sqrt(bsize))
        samples_grid = [
            make_grid(samples[i * bsize:(i + 1) * bsize], nrow, padding=2)[None, ...] for i in range(len(frames))
        ]
        samples_grid = torch.cat(samples_grid, 0)

        clean_grid = make_grid(batch, nrow, padding=2)[None, ...]
        noisy_grid = make_grid(noisy_batch, nrow, padding=2)[None, ...]
        # Write to file
        clean_grid = clean_grid.permute(0, 2, 3, 1)
        noisy_grid = noisy_grid.permute(0, 2, 3, 1)
        samples_grid = samples_grid.permute(0, 2, 3, 1)
        dump_data(clean_grid, noisy_grid, samples_grid, this_sample_dir, 0)
        gc.collect()


if __name__ == '__main__':

    app.run(main)
