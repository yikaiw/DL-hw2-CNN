import numpy as np
import tensorflow as tf
import os
import sys
import config
import time


class tfrecord_read(object):
    def __init__(self, dataset_name='dset1', batch_size=config.batch_size,
                 num_epochs=config.num_epochs, train_slice=0.8, training=True):
        # Data has been shuffled manually before.
        # Take first train_slice files for training, and take last 1-train_slice files for validation.
        tfrecord_dir = os.path.join('tfrecord', dataset_name)
        tfrecord_files = os.listdir(tfrecord_dir)
        tfrecord_files = list(map(lambda s: os.path.join(tfrecord_dir, s), tfrecord_files))
        file_num = len(tfrecord_files)
        slice_pos = int(np.ceil(file_num * train_slice))
        if training:
            tfrecord_files = tfrecord_files[:slice_pos]
        else:
            tfrecord_files = tfrecord_files[slice_pos:]
        filename_queue = tf.train.string_input_producer(tfrecord_files, num_epochs=num_epochs, shuffle=True)

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized=serialized_example,
            features={'image': tf.FixedLenFeature([], tf.string), 'label': tf.FixedLenFeature([], tf.int64)})

        img = features['image']
        img = tf.image.decode_png(img, channels=3)  # get tf.Tensor([height, width, channel], dtype=float32)
        img = tf.image.resize_images(img, size=[config.img_size, config.img_size])
        label = features['label']

        self.X_batch, self.y_batch = tf.train.shuffle_batch(
            tensors=[img, label], batch_size=batch_size, capacity=5000, min_after_dequeue=100, num_threads=3)


def test_reader():
    reader = tfrecord_read()  # must be defined before sess, or tf will stuck at sess.run

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    X_train_batch, y_train_batch = sess.run([reader.X_batch, reader.y_batch])
    print(X_train_batch.shape, y_train_batch.shape)

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_reader()