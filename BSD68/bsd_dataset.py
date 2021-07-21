import os

import numpy as np
import tensorflow as tf


def get_preprocess_fn(crop_size, random_flip, evaluation):
    """
  Take a random crop
  """
    def preprocess_fn(img_data):
        both_noisy_and_clean = set(img_data.keys()) == set(['image', 'ground_truth'])
        if both_noisy_and_clean:
            img = tf.concat([img_data['image'][None, ...], img_data['ground_truth'][None, ...]], 0)
            img = tf.image.random_crop(value=img, size=(2, crop_size, crop_size))

        else:
            img = img_data['image']
            img = tf.image.random_crop(value=img, size=(crop_size, crop_size))

        img = tf.expand_dims(img, -1)
        # img = img / 255.0
        img = tf.clip_by_value(img, clip_value_min=0, clip_value_max=255) / 255.0

        if random_flip and not evaluation:
            img = tf.image.random_flip_left_right(img)

        if both_noisy_and_clean:
            return {'image': img[0], 'ground_truth': img[1]}
        else:
            return {'image': img}

    return preprocess_fn


def set_dataset_options(ds, config, evaluation):
    shuffle_buffer_size = 1000
    prefetch_size = tf.data.experimental.AUTOTUNE
    num_epochs = None if not evaluation else 1
    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
    crop_size = config.data.image_size

    ds = ds.map(get_preprocess_fn(crop_size, config.data.random_flip, evaluation),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.repeat(count=num_epochs)
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)


def get_bsd_dataset(split, config, evaluation=False, data_dir='/tmp2/ashesh/ashesh/BSD68_reproducibility_data/'):
    if split == 'train':
        data = np.load(os.path.join(data_dir, 'train/DCNN400_train_gaussian25.npy'), allow_pickle=True)
        dset = tf.data.Dataset.from_tensor_slices({'image': data})
    elif split == 'val':
        data = np.load(os.path.join(data_dir, 'val/DCNN400_validation_gaussian25.npy'), allow_pickle=True)
        n = int(np.ceil(config.data.raw_img_size / config.data.image_size))
        data = np.repeat(data, n**2, axis=0)
        dset = tf.data.Dataset.from_tensor_slices({'image': data})
    elif split == 'test':

        def generator():
            data = np.load(os.path.join(data_dir, 'test/bsd68_gaussian25.npy'), allow_pickle=True)
            truth = np.load(os.path.join(data_dir, 'test/bsd68_groundtruth.npy'), allow_pickle=True)

            truth = [x if x.shape[0] == 321 else x.T for x in truth]
            data = [x if x.shape[0] == 321 else x.T for x in data]
            for s_d, s_t in zip(data, truth):
                yield {"image": s_d, "ground_truth": s_t}

        dset = tf.data.Dataset.from_generator(generator,
                                              output_types=({
                                                  "image": tf.float32,
                                                  "ground_truth": tf.float32
                                              }))
    return set_dataset_options(dset, config, evaluation)
