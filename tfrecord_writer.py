import numpy as np
import tensorflow as tf
import os
import sys
import time
from tqdm import tqdm
import config

dset_dirs = ['data/dset1/train', 'data/dset2/train']
tfrecord_dirs = ['tfrecord/dset1/', 'tfrecord/dset2/']
tfrecord_num = config.tfrecord_num  # num of tfrecord files


def get_file_path(data_path):
    # get the path and label for every .jpg file
    img_paths, labels = [], []
    class_dirs = sorted(os.listdir(data_path))
    dict_class2id = {}  # for matching the classes between test data and train data
    
    for i in range(len(class_dirs)):
        label = i
        class_dir = class_dirs[i]
        dict_class2id[class_dir] = label
        class_path = os.path.join(data_path, class_dir)
        file_names = sorted(os.listdir(class_path))

        for file_name in file_names:
            file_path = os.path.join(class_path, file_name)
            img_paths.append(file_path)
            labels.append(label)

    img_paths = np.asarray(img_paths)
    labels = np.asarray(labels)
    return img_paths, labels


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))


def convert_tfrecord_dataset(img_paths, labels, tfrecord_dir, tfrecord_num, shuffle=True):
    if not os.path.exists(tfrecord_dir):
        os.makedirs(tfrecord_dir)
    n_sample = len(img_paths)
    num_per_shard = n_sample // tfrecord_num  # samples number of each tfrecord

    # shuffle manually
    if shuffle:
        new_idxs = np.random.permutation(n_sample)
        img_paths = img_paths[new_idxs]
        labels = labels[new_idxs]

    time0 = time.time()
    for shard_id in range(tfrecord_num):
        # output_filename = '%d-of-%d.tfrecord' % (shard_id, n_shards)
        output_filename = str(shard_id).zfill(len(str(tfrecord_num))) + '.tfrecord'
        output_path = os.path.join(tfrecord_dir, output_filename)

        with tf.python_io.TFRecordWriter(output_path) as writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id + 1) * num_per_shard, n_sample)
            for i in range(start_ndx, end_ndx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d, %g s' % (
                    i + 1, n_sample, shard_id, time.time() - time0))
                sys.stdout.flush()
                png_path = img_paths[i]
                label = labels[i]
                img = tf.gfile.FastGFile(png_path, 'rb').read()  # read image
                # tf.image.decode_jpeg(img)
                example = tf.train.Example(features=tf.train.Features(
                    feature={'image': bytes_feature(img), 'label': int64_feature(label)}))
                writer.write(example.SerializeToString())


if __name__ == '__main__':
    for i in range(len(dset_dirs)):
        img_paths, labels = get_file_path(data_path=dset_dirs[i])
        convert_tfrecord_dataset(img_paths, labels, tfrecord_dirs[i], tfrecord_num, shuffle=True)
    print('\nFinished writing data to tfrecord files.')