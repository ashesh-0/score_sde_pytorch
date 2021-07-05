import csv
import functools
import importlib
import inspect
import io
import itertools
import os
from dataclasses import dataclass, field

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import torch
import torch.nn as nn
import tqdm

sns.set(font_scale=2)
sns.set(style="whitegrid")

import models
#from configs.ncsnpp import cifar10_continuous_ve as configs
from configs.ddpm import cifar10_continuous_vp as configs
from models import ddpm as ddpm_model
from models import layers, layerspp, ncsnpp, ncsnv2, normalization
from models import utils as mutils

config = configs.get_config()

checkpoint = torch.load('exp/ddpm_continuous_vp.pth')

#score_model = ncsnpp.NCSNpp(config)
score_model = ddpm_model.DDPM(config)
score_model.load_state_dict(checkpoint)
score_model = score_model.eval()
x = torch.ones(8, 3, 32, 32)
y = torch.tensor([1] * 8)
breakpoint()
with torch.no_grad():
    score = score_model(x, y)
