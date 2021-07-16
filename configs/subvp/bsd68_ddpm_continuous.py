# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training DDPM with sub-VP SDE."""

from configs.default_cifar10_configs import get_default_configs


def get_config():
    config = get_default_configs()

    # training
    training = config.training
    training.sde = 'subvpsde'
    training.continuous = True
    training.reduce_mean = True
    training.start_t = 0.1
    # training.batch_size = 32
    # sampling
    sampling = config.sampling
    sampling.method = 'ode'
    sampling.predictor = 'euler_maruyama'
    sampling.corrector = 'none'

    # data
    data = config.data
    data.dataset = 'BSD68'
    data.image_size = 180
    data.random_flip = True
    data.centered = False
    data.uniform_dequantization = False
    data.num_channels = 3
    data.base_noise_std = 25
    data.crop_size = 128

    # model
    model = config.model
    model.name = 'ddpm'
    model.scale_by_sigma = False
    model.ema_rate = 0.9999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 2
    model.attn_resolutions = (16, )
    model.resamp_with_conv = True
    model.conditional = True

    # Evaluate
    eval = config.eval
    eval.begin_ckpt = 8
    eval.end_ckpt = 8
    eval.denoising_samples = False
    eval.enable_sampling = True
    eval.batch_size = 128
    eval.num_samples = 1024
    return config
