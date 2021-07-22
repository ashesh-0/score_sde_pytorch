"""
Here, one can get reconstruction of the images.
We save three set of files:
1. Clean images.
2. Reconstructed images.
3. Noisy images (which were present in the training.)
"""
import gc
import logging
import os

import numpy as np
import tensorflow as tf
import torch
from absl import app, flags
from denoise_utils import Setup, write_to_file
from eval_metrics import EvalMetrics
from ml_collections.config_flags import config_flags
# Keep the import below for registering all model definitions
from tqdm import tqdm

from scripts.get_marginal_probablity import (convert_to_img, get_noisy_imgs_from_batch)

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")

flags.DEFINE_float('latent_start_t', 1e-3, 'Latent start_t')
flags.DEFINE_float('sampling_end_t', 1e-3, 'Sampling end_t')
flags.DEFINE_bool('skip_forward_integration', False, 'If True, the the forward integration is not used.'
                  ' We just use the reverse integration')

# flags.DEFINE_string("eval_folder", "eval", "The folder name for storing evaluation results")
flags.mark_flags_as_required(["workdir", "config"])

FLAGS = flags.FLAGS


def dump_data(batch, noisy_batch, samples, output_dir, sampling_round):
    samples = (samples.cpu().numpy()).astype(np.uint8)
    batch = (batch.cpu().numpy()).astype(np.uint8)
    noisy_batch = (noisy_batch.cpu().numpy()).astype(np.uint8)

    # Write samples to disk or Google Cloud Storage
    write_to_file(os.path.join(output_dir, f"samples_{sampling_round}.npz"), samples)
    write_to_file(os.path.join(output_dir, f"clean_data_{sampling_round}.npz"), batch)
    write_to_file(os.path.join(output_dir, f"noisy_data_{sampling_round}.npz"), noisy_batch)


def post_process_data(setup, batch, noisy_batch, samples):
    samples = samples.permute(0, 2, 3, 1)
    samples = torch.clamp(samples, min=0, max=1)
    samples = 255 * samples

    noisy_batch = convert_to_img(noisy_batch, setup.inverse_scaler)
    noisy_batch = 255 * noisy_batch

    batch = setup.inverse_scaler(batch).permute(0, 2, 3, 1)
    batch = 255 * batch
    return batch, noisy_batch, samples


def main(argv):
    """
    latent_start_t: For computing latent representation, what do we want to say about the start time.
    noise_start_t: How much noise do we want to add to the data. This noisy data will then be used as starting point
                    for denoising.
    sampling_end_t: In backward ode integration used for sampling, till what time do we want to integrate back.
    """
    workdir = FLAGS.workdir
    latent_start_t = FLAGS.latent_start_t
    sampling_end_t = FLAGS.sampling_end_t
    config = FLAGS.config
    start_t = config.training.start_t
    skip_forward_integration = FLAGS.skip_forward_integration

    eval_dir = (f'{workdir}/eval_dir_nt_{start_t}_'
                f'lt_{latent_start_t}_st_{sampling_end_t}_skipF_{int(skip_forward_integration)}')
    print('')
    print('Evaluation will be saved to ', eval_dir)
    print('')

    setup = Setup(config, workdir)
    tf.io.gfile.makedirs(eval_dir)

    input_metrics = EvalMetrics('Input')
    recons_metrics = EvalMetrics('Recons')

    train_iter = iter(setup.eval_ds)
    # checkpoint_dir = os.path.join(workdir, "checkpoints")
    begin_ckpt = config.eval.begin_ckpt
    logging.info("begin checkpoint: %d" % (begin_ckpt, ))
    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
        setup.load_weights(ckpt)

        assert config.eval.enable_sampling
        num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1

        this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
        tf.io.gfile.makedirs(this_sample_dir)
        for r in tqdm(range(num_sampling_rounds)):
            logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

            batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
            batch = batch.permute(0, 3, 1, 2)
            batch = setup.scaler(batch)

            noisy_batch = get_noisy_imgs_from_batch(start_t - setup.existing_noise_t, batch, setup.sde,
                                                    setup.inverse_scaler)
            if skip_forward_integration:
                samples, n = setup.sampling_fn(
                    setup.score_model,
                    z=noisy_batch,
                    end_t=sampling_end_t,
                    start_t=latent_start_t,
                )

            else:
                _, z, _ = setup.likelihood_fn(setup.score_model, noisy_batch, start_t=latent_start_t)

                samples, n = setup.sampling_fn(
                    setup.score_model,
                    z=z,
                    end_t=sampling_end_t,
                )

            # Convert them to [0,255] and uniform shape
            batch, noisy_batch, samples = post_process_data(setup, batch, noisy_batch, samples)

            input_metrics.update(batch, noisy_batch)
            recons_metrics.update(batch, samples)

            # Write to file
            dump_data(batch, noisy_batch, samples, this_sample_dir, r)
            gc.collect()

    print('')
    print('------------------------------------------------')
    print('')
    _ = input_metrics.get()
    _ = recons_metrics.get()


if __name__ == '__main__':

    app.run(main)
