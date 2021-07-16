import os

import numpy as np
import tensorflow as tf


def get_preprocess_fn(crop_size, random_flip, evaluation):
    """
  Take a random crop
  """
    def preprocess_fn(img):
        img = tf.image.random_crop(value=img, size=(crop_size, crop_size))
        img = tf.expand_dims(img, -1)
        if random_flip and not evaluation:
            img = tf.image.random_flip_left_right(img)
        return img

    return preprocess_fn


def set_dataset_options(ds, config, evaluation):
    shuffle_buffer_size = 1000
    prefetch_size = tf.data.experimental.AUTOTUNE
    num_epochs = None if not evaluation else int(np.ceil(config.image_size / config.crop_size))
    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
    crop_size = config.data.crop_size

    ds = ds.map(get_preprocess_fn(crop_size, config.data.random_flip, evaluation),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.repeat(count=num_epochs)
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)


def get_bsd_dataset(split, config, data_dir='/tmp2/ashesh/ashesh/BSD68_reproducibility_data/'):
    if split == 'train':
        data = np.load(os.path.join(data_dir, 'train/DCNN400_train_gaussian25.npy'), allow_pickle=True)[:1]
    elif split == 'val':
        data = np.load(os.path.join(data_dir, 'train/DCNN400_validation_gaussian25.npy'), allow_pickle=True)

    dset = tf.data.Dataset.from_tensor_slices(data)
    evaluation = split != 'train'
    return set_dataset_options(dset, config, evaluation)
