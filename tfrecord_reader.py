import numpy as np
import tensorflow as tf
import os
import sys
import config
import time


def tfrecord_read(dataset_name='dset1', training=True, train_slice=0.8):
    # Data has been shuffled manually before.
    # Take first train_slice files for training, and take last 1-train_slice files for validation.
    tfrecord_dir = os.path.join('tfrecord', dataset_name)
    tfrecord_files = os.listdir(tfrecord_dir)
    tfrecord_files = list(map(lambda s: os.path.join(tfrecord_dir, s), tfrecord_files))
    file_num = len(tfrecord_files)
    if training:
        tfrecord_files = tfrecord_files[:np.ceil(file_num * train_slice)]
    else:
        tfrecord_files = tfrecord_files[np.ceil(file_num * train_slice):]
    filename_queue = tf.train.string_input_producer(tfrecord_files, num_epochs=None, shuffle=True)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized=serialized_example, 
        features={'image': tf.FixedLenFeature([], tf.string), 'label': tf.FixedLenFeature([], tf.int64)})

    img = features['image']
    img = tf.image.decode_png(img, channels=3)  # get tf.Tensor([height, width, channel], dtype=float32)
    img = tf.image.resize_images(img, size=[config.img_size, config.img_size])
    label = features['label']

    X_batch, y_batch = tf.train.shuffle_batch(
        tensors=[img, label], batch_size=config.batch_size, capacity=5000, min_after_dequeue=100, num_threads=3)

    return X_batch, y_batch