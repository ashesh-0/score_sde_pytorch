# Score-Based Generative Modeling through Stochastic Differential Equations

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/score-based-generative-modeling-through-1/image-generation-on-cifar-10)](https://paperswithcode.com/sota/image-generation-on-cifar-10?p=score-based-generative-modeling-through-1)

This repo contains a PyTorch implementation for the paper [Score-Based Generative Modeling through Stochastic Differential Equations](https://openreview.net/forum?id=PxTIG12RRHS)

by [Yang Song](https://yang-song.github.io), [Jascha Sohl-Dickstein](http://www.sohldickstein.com/), [Diederik P. Kingma](http://dpkingma.com/), [Abhishek Kumar](http://users.umiacs.umd.edu/~abhishek/), [Stefano Ermon](https://cs.stanford.edu/~ermon/), and [Ben Poole](https://cs.stanford.edu/~poole/)

--------------------

We propose a unified framework that generalizes and improves previous work on score-based generative models through the lens of stochastic differential equations (SDEs). In particular, we can transform data to a simple noise distribution with a continuous-time stochastic process described by an SDE. This SDE can be reversed for sample generation if we know the score of the marginal distributions at each intermediate time step, which can be estimated with score matching. The basic idea is captured in the figure below:

![schematic](assets/schematic.jpg)

Our work enables a better understanding of existing approaches,  new sampling algorithms, exact likelihood computation, uniquely identifiable encoding, latent code manipulation, and brings new conditional generation abilities (including but not limited to class-conditional generation, inpainting and colorization) to the family of score-based generative models.

All combined, we achieved an FID of **2.20** and an Inception score of **9.89** for unconditional generation on CIFAR-10, as well as high-fidelity generation of **1024px** Celeba-HQ images (samples below). In addition, we obtained a likelihood value of **2.99** bits/dim on uniformly dequantized CIFAR-10 images.

![FFHQ samples](assets/ffhq_samples.jpg)

## What does this code do?
Aside from the **NCSN++** and **DDPM++** models in our paper, this codebase also re-implements many previous score-based models in one place, including **NCSN** from [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600), **NCSNv2** from [Improved Techniques for Training Score-Based Generative Models](https://arxiv.org/abs/2006.09011), and **DDPM** from [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).

It supports training new models, evaluating the sample quality and likelihoods of existing models. We carefully designed the code to be modular and easily extensible to new SDEs, predictors, or correctors.

## JAX version

Please find a JAX implementation [here](https://github.com/yang-song/score_sde), which additionally supports class-conditional generation with a pre-trained classifier, and resuming an evalution process after pre-emption.

###  JAX vs. PyTorch

In general, this PyTorch version consumes less memory but runs slower than JAX. Here is a benchmark on training an NCSN++ cont. model with VE SDE. Hardware is 4x Nvidia Tesla V100 GPUs (32GB)
| Framework | Time (second per step) | Memory usage in total (GB) |
|:----:|:----:|:----:|
|PyTorch | 0.56 | 20.6|
|JAX (`n_jitted_steps=1`)| 0.30 | 29.7 |
|JAX (`n_jitted_steps=5`) | 0.20 | 74.8|

## How to run the code

### Dependencies

Run the following to install a subset of necessary python packages for our code
```sh
pip install -r requirements.txt
```

### Stats files for quantitative evaluation

We provide the stats file for CIFAR-10. You can download [`cifar10_stats.npz`](https://drive.google.com/file/d/14UB27-Spi8VjZYKST3ZcT8YVhAluiFWI/view?usp=sharing)  and save it to `assets/stats/`. Check out [#5](https://github.com/yang-song/score_sde/pull/5) on how to compute this stats file for new datasets.

### Usage

Train and evaluate our models through `main.py`.

```sh
main.py:
  --config: Training configuration.
    (default: 'None')
  --eval_folder: The folder name for storing evaluation results
    (default: 'eval')
  --mode: <train|eval>: Running mode: train or eval
  --workdir: Working directory
```

* `config` is the path to the config file. Our prescribed config files are provided in `configs/`. They are formatted according to [`ml_collections`](https://github.com/google/ml_collections) and should be quite self-explanatory.

  **Naming conventions of config files**: the path of a config file is a combination of the following dimensions:
  *  dataset: One of `cifar10`, `celeba`, `celebahq`, `celebahq_256`, `ffhq_256`, `celebahq`, `ffhq`.
  * model: One of `ncsn`, `ncsnv2`, `ncsnpp`, `ddpm`, `ddpmpp`.
  * continuous: train the model with continuously sampled time steps.

*  `workdir` is the path that stores all artifacts of one experiment, like checkpoints, samples, and evaluation results.

* `eval_folder` is the name of a subfolder in `workdir` that stores all artifacts of the evaluation process, like meta checkpoints for pre-emption prevention, image samples, and numpy dumps of quantitative results.

* `mode` is either "train" or "eval". When set to "train", it starts the training of a new model, or resumes the training of an old model if its meta-checkpoints (for resuming running after pre-emption in a cloud environment) exist in `workdir/checkpoints-meta` . When set to "eval", it can do an arbitrary combination of the following

  * Evaluate the loss function on the test / validation dataset.

  * Generate a fixed number of samples and compute its Inception score, FID, or KID. Prior to evaluation, stats files must have already been downloaded/computed and stored in `assets/stats`.

  * Compute the log-likelihood on the training or test dataset.

  These functionalities can be configured through config files, or more conveniently, through the command-line support of the `ml_collections` package. For example, to generate samples and evaluate sample quality, supply the  `--config.eval.enable_sampling` flag; to compute log-likelihoods, supply the `--config.eval.enable_bpd` flag, and specify `--config.eval.dataset=train/test` to indicate whether to compute the likelihoods on the training or test dataset.

## How to extend the code
* **New SDEs**: inherent the `sde_lib.SDE` abstract class and implement all abstract methods. The `discretize()` method is optional and the default is Euler-Maruyama discretization. Existing sampling methods and likelihood computation will automatically work for this new SDE.
* **New predictors**: inherent the `sampling.Predictor` abstract class, implement the `update_fn` abstract method, and register its name with `@register_predictor`. The new predictor can be directly used in `sampling.get_pc_sampler` for Predictor-Corrector sampling, and all other controllable generation methods in `controllable_generation.py`.
* **New correctors**: inherent the `sampling.Corrector` abstract class, implement the `update_fn` abstract method, and register its name with `@register_corrector`. The new corrector can be directly used in `sampling.get_pc_sampler`, and all other controllable generation methods in `controllable_generation.py`.

## Pretrained checkpoints
All checkpoints are provided in this [Google drive](https://drive.google.com/drive/folders/1tFmF_uh57O6lx9ggtZT_5LdonVK2cV-e?usp=sharing).

**Instructions**: You may find two checkpoints for some models. The first checkpoint (with a smaller number) is the one that we reported FID scores in our paper's Table 3 (also corresponding to the FID and IS columns in the table below). The second checkpoint (with a larger number) is the one that we reported likelihood values and FIDs of black-box ODE samplers in our paper's Table 2 (also FID(ODE) and NNL (bits/dim) columns in the table below). The former corresponds to the smallest FID during the course of training (every 50k iterations). The later is the last checkpoint during training.

Per Google's policy, we cannot release our original CelebA and CelebA-HQ checkpoints. That said, I have re-trained models on FFHQ 1024px, FFHQ 256px and CelebA-HQ 256px with personal resources, and they achieved similar performance to our internal checkpoints.

Here is a detailed list of checkpoints and their results reported in the paper. **FID (ODE)** corresponds to the sample quality of black-box ODE solver applied to the probability flow ODE.

| Checkpoint path | FID | IS | FID (ODE) | NNL (bits/dim) |
|:----------|:-------:|:----------:|:----------:|:----------:|
| [`ve/cifar10_ncsnpp/`](https://drive.google.com/drive/folders/1sP4GwvrYiI-sDPTp7sKYzsxJLGVamVMZ?usp=sharing) |  2.45 | 9.73 | - | - |
| [`ve/cifar10_ncsnpp_continuous/`](https://drive.google.com/drive/folders/1b0gy_LLgO_DaQBgoWXwlVnL_rcAUgREh?usp=sharing) | 2.38 | 9.83 | - | - |
| [`ve/cifar10_ncsnpp_deep_continuous/`](https://drive.google.com/drive/folders/11s6A_xM7qiztdj8AHQWqaIAUSC3I7uX2?usp=sharing) | **2.20** | **9.89** | - | - |
| [`vp/cifar10_ddpm/`](https://drive.google.com/drive/folders/1zDKcy3xbsN3F4AfyB_DfY_1oho89iKcf?usp=sharing) | 3.24 | - | 3.37 | 3.28 |
| [`vp/cifar10_ddpm_continuous`](https://drive.google.com/drive/folders/1RHNxW1qY-mTr0JMAE5t4V181Hi_aVWXK?usp=sharing) | - | - | 3.69| 3.21 |
| [`vp/cifar10_ddpmpp`](https://drive.google.com/drive/folders/1zOVj03ZBcq339p5QEKJPh2bBrxR_HOCM?usp=sharing) | 2.78 | 9.64 | - | - |
| [`vp/cifar10_ddpmpp_continuous`](https://drive.google.com/drive/folders/1xYjVMx10N9ivQQBIsEoXEeu9nvSGTBrC?usp=sharing) | 2.55 | 9.58 | 3.93 | 3.16 |
| [`vp/cifar10_ddpmpp_deep_continuous`](https://drive.google.com/drive/folders/1ZMLBiu9j7-rpdTQu8M2LlHAEQq4xRYrj?usp=sharing) | 2.41 | 9.68 | 3.08 | 3.13 |
| [`subvp/cifar10_ddpm_continuous`](https://drive.google.com/drive/folders/1DeebpmBkCxlZx89t3z45Te37T7BPOzd2?usp=sharing) | - | - | 3.56 | 3.05 |
| [`subvp/cifar10_ddpmpp_continuous`](https://drive.google.com/drive/folders/1bLgmnEAZnysRZfWt8qN3omGfijJ_B884?usp=sharing) | 2.61 | 9.56 | 3.16 | 3.02 |
| [`subvp/cifar10_ddpmpp_deep_continuous`](https://drive.google.com/drive/folders/16QGkviGcizSbIPRk37-YksUhlNIna4Ys?usp=sharing) | 2.41 | 9.57 | **2.92** | **2.99** |

| Checkpoint path | Samples |
|:-----|:------:|
| [`ve/bedroom_ncsnpp_continuous`](https://drive.google.com/drive/folders/18GmxDvfGR8se9uFucc9uweeVrX_GzuUG?usp=sharing) | ![bedroom_samples](assets/bedroom.jpeg) |
| [`ve/church_ncsnpp_continuous`](https://drive.google.com/drive/folders/1zVChA0HrnJU66Jkt4P6KOnlREhBMc4Yh?usp=sharing) | ![church_samples](assets/church.jpeg) |
| [`ve/ffhq_1024_ncsnpp_continuous`](https://drive.google.com/drive/folders/1ZqLNr_kH0o9DxvwSlrQPMmkrhEnXhBm2?usp=sharing) |![ffhq_1024](assets/ffhq_1024.jpeg)|
| [`ve/ffhq_256_ncsnpp_continuous`](https://drive.google.com/drive/folders/1KG72ZKUCUa8dDcA03hOf1BsnK8kBcdPD?usp=sharing) |![ffhq_256_samples](assets/ffhq_256.jpg)|
| [`ve/celebahq_256_ncsnpp_continuous`](https://drive.google.com/drive/folders/19VJ7UZTE-ytGX6z5rl-tumW9c0Ps3itk?usp=sharing) |![celebahq_256_samples](assets/celebahq_256.jpg)|


## Demonstrations and tutorials
| Link | Description|
|:----:|:-----|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dRR_0gNRmfLtPavX2APzUggBuXyjWW55?usp=sharing)  | Load our pretrained checkpoints and play with sampling, likelihood computation, and controllable synthesis (JAX + FLAX)|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17lTrPLTt_0EDXa4hkbHmbAFQEkpRDZnh?usp=sharing) | Load our pretrained checkpoints and play with sampling, likelihood computation, and controllable synthesis (PyTorch) |
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SeXMpILhkJPjXUaesvzEhc3Ke6Zl_zxJ?usp=sharing) | Tutorial of score-based generative models in JAX + FLAX |
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing)| Tutorial of score-based generative models in PyTorch |


## Tips
* When using the JAX codebase, you can jit multiple training steps together to improve training speed at the cost of more memory usage. This can be set via `config.training.n_jitted_steps`. For CIFAR-10, we recommend using `config.training.n_jitted_steps=5` when your GPU/TPU has sufficient memory; otherwise we recommend using `config.training.n_jitted_steps=1`. Our current implementation requires `config.training.log_freq` to be dividable by `n_jitted_steps` for logging and checkpointing to work normally.
* The `snr` (signal-to-noise ratio) parameter of `LangevinCorrector` somewhat behaves like a temperature parameter. Larger `snr` typically results in smoother samples, while smaller `snr` gives more diverse but lower quality samples. Typical values of `snr` is `0.05 - 0.2`, and it requires tuning to strike the sweet spot.
* For VE SDEs, we recommend choosing `config.model.sigma_max` to be the maximum pairwise distance between data samples in the training dataset.

## References

If you find the code useful for your research, please consider citing
```bib
@inproceedings{
  song2021scorebased,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Yang Song and Jascha Sohl-Dickstein and Diederik P Kingma and Abhishek Kumar and Stefano Ermon and Ben Poole},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=PxTIG12RRHS}
}
```

This work is built upon some previous papers which might also interest you:

* Song, Yang, and Stefano Ermon. "Generative Modeling by Estimating Gradients of the Data Distribution." *Proceedings of the 33rd Annual Conference on Neural Information Processing Systems*. 2019.
* Song, Yang, and Stefano Ermon. "Improved techniques for training score-based generative models." *Proceedings of the 34th Annual Conference on Neural Information Processing Systems*. 2020.
* Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." *Proceedings of the 34th Annual Conference on Neural Information Processing Systems*. 2020.


[EvalMetrics Input] MSE:184.18+-5.848
[EvalMetrics Input] PSNR:25.48+-0.138

### How much back should one go while doing sampling?

--noise_start_t=0.1 --latent_start_t=0.1 --sampling_start_t=0.1
[EvalMetrics Recons] MSE:184.90+-16.415
[EvalMetrics Recons] PSNR:25.47+-0.258

--noise_start_t=0.1 --latent_start_t=0.1 --sampling_start_t=0.09
[EvalMetrics Recons] MSE:142.70+-14.229
[EvalMetrics Recons] PSNR:26.60+-0.300

--noise_start_t=0.1 --latent_start_t=0.1 --sampling_start_t=0.05
[EvalMetrics Recons] MSE:85.13+-42.172
[EvalMetrics Recons] PSNR:28.92+-0.614

--noise_start_t=0.1 --latent_start_t=0.1 --sampling_start_t=0.001
[EvalMetrics Recons] MSE:75.25+-58.295
[EvalMetrics Recons] PSNR:29.53+-0.843

--noise_start_t=0.1 --latent_start_t=0.1 --sampling_start_t=-0.05
[EvalMetrics Recons] MSE:72.18+-12.587
[EvalMetrics Recons] PSNR:29.59+-0.578

--noise_start_t=0.1 --latent_start_t=0.1 --sampling_start_t=-0.1
[EvalMetrics Recons] MSE:72.17+-8.731
[EvalMetrics Recons] PSNR:29.58+-0.486

--noise_start_t=0.1 --latent_start_t=0.1 --sampling_start_t=-0.15
[EvalMetrics Recons] MSE:72.63+-15.301
[EvalMetrics Recons] PSNR:29.57+-0.622

--noise_start_t=0.1 --latent_start_t=0.1 --sampling_start_t=-0.3
[EvalMetrics Recons] MSE:73.47+-27.320
[EvalMetrics Recons] PSNR:29.56+-0.708

--noise_start_t=0.1 --latent_start_t=0.1 --sampling_start_t=-0.5
[EvalMetrics Recons] MSE:73.81+-35.179
[EvalMetrics Recons] PSNR:29.56+-0.743


### Trying out denoising different amounts of noise


### Latent representation start_t different from actual
--noise_start_t=0.1 --latent_start_t=0.08
[EvalMetrics Recons] MSE:102.19+-21.287
[EvalMetrics Recons] PSNR:28.08+-0.523

--noise_start_t=0.1 --latent_start_t=0.11
[EvalMetrics Recons] MSE:63.66+-30.777
[EvalMetrics Recons] PSNR:30.25+-0.952

--noise_start_t=0.1 --latent_start_t=0.12
[EvalMetrics Recons] MSE:57.47+-23.756
[EvalMetrics Recons] PSNR:30.68+-1.002

--noise_start_t=0.1 --latent_start_t=0.125
[EvalMetrics Recons] MSE:57.60+-32.041
[EvalMetrics Recons] PSNR:30.72+-1.102

--noise_start_t=0.1 --latent_start_t=0.13
[EvalMetrics Recons] MSE:57.71+-25.711
[EvalMetrics Recons] PSNR:30.69+-1.095

--noise_start_t=0.1 --latent_start_t=0.15
[EvalMetrics Recons] MSE:70.06+-22.319
[EvalMetrics Recons] PSNR:29.84+-1.155

--noise_start_t=0.1 --latent_start_t=0.2
[EvalMetrics Recons] MSE:155.77+-44.643
[EvalMetrics Recons] PSNR:26.40+-1.370

### noise and latent representation have same start_t which is different from 0.1
--noise_start_t=0.05 --latent_start_t=0.05 --sampling_start_t=0.001
[EvalMetrics Input] MSE:14.92+-0.508
[EvalMetrics Input] PSNR:36.40+-0.147
[EvalMetrics Recons] MSE:14.82+-29.564
[EvalMetrics Recons] PSNR:37.05+-1.328

--noise_start_t=0.15 --latent_start_t=0.15 --sampling_start_t=0.001
[EvalMetrics Input] MSE:763.05+-21.152
[EvalMetrics Input] PSNR:19.31+-0.120
[EvalMetrics Recons] MSE:136.06+-30.818
[EvalMetrics Recons] PSNR:26.90+-0.947

--noise_start_t=0.2 --latent_start_t=0.2 --sampling_start_t=-0.05
[EvalMetrics Input] MSE:1925.68+-57.070
[EvalMetrics Input] PSNR:15.29+-0.128
[EvalMetrics Recons] MSE:236.29+-59.675
[EvalMetrics Recons] PSNR:24.54+-1.142

--noise_start_t=0.2 --latent_start_t=0.2 --sampling_start_t=0.0
[EvalMetrics Recons] MSE:238.18+-59.125
[EvalMetrics Recons] PSNR:24.50+-1.133

### Skipping the forward integration.
Performance is slightly better. And the time is reduced by almost half.
--noise_start_t=0.1 --latent_start_t=0.1 --sampling_end_t=-0.05 --skip_forward_integration
[EvalMetrics Recons] MSE:71.17+-7.271
[EvalMetrics Recons] PSNR:29.63+-0.450
