import tensorflow as tf
import os
import sys
import config

def tfrecord_read(dataset_name='dset1'):
    tfrecord_dir = os.path.join('tfrecord', dataset_name)
    tfrecord_files = os.listdir(tfrecord_dir)
    tfrecord_files = list(map(lambda s: os.path.join(tfrecord_dir, s), tfrecord_files))
    filename_queue = tf.train.string_input_producer(tfrecord_files, num_epochs=None, shuffle=True)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized=serialized_example, 
        features={'image': tf.FixedLenFeature([], tf.string), 'label': tf.FixedLenFeature([], tf.int64)})

    img = features['image']
    img = tf.image.decode_png(img, channels=3)
    img = tf.reshape(img, [config.imgsize, config.imgsize, 3])
    label = features['label']

    X_batch, y_batch = tf.train.shuffle_batch(
        tensors=[img, label], batch_size=config.batch_size, capacity=5000, min_after_dequeue=100, num_threads=3)

    # return X_batch, y_batch
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    y_outputs = list()
    for i in range(5):
        _X_batch, _y_batch = sess.run([X_batch, y_batch])
        print('** batch %d' % i)
        print('_X_batch.shape:', _X_batch.shape)
        print('_y_batch:', _y_batch)
        y_outputs.extend(_y_batch.tolist())
    print(y_outputs)

    time0 = time.time()
    for count in range(100):  # 100 batch, 5.4 seconds
        _X_batch, _y_batch = sess.run([X_batch, y_batch])
        sys.stdout.write("\rloop {}, pass {:.2f}s".format(count, time.time() - time0))
        sys.stdout.flush()

    coord.request_stop()
    coord.join(threads)

tfrecord_read()