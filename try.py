import tensorflow as tf
from structure import CNN
from datetime import datetime
import os
from tfrecord_reader import tfrecord_read
import config


def main():
    cnn = CNN()
    read_for_train = tfrecord_read(
        'dset1', config.batch_size, config.num_epochs, config.train_slice, training=True)
    read_for_val = tfrecord_read(
        'dset1', config.batch_size, config.num_epochs, config.train_slice, training=False)


    print('Build session.')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        step = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            print('Start training:')
            while not coord.should_stop():
                X_train_batch, y_train_batch = sess.run([read_for_train.X_batch, read_for_train.y_batch])
                # print(X_train_batch.shape, y_train_batch.shape)

                loss = sess.run([cnn.loss],
                    {cnn.X_inputs: X_train_batch, cnn.y_inputs: y_train_batch, cnn.training: True})
                #
                # X_val_batch, y_val_batch = sess.run([read_for_val.X_batch, read_for_val.y_batch])
                # accuracy = sess.run(cnn.accuracy,
                #     {cnn.X_inputs: X_val_batch, cnn.y_inputs: y_val_batch, cnn.training: False})
                #
                if step % 100 == 0:
                    print('At step {}:'.format(step))
                    print('\tloss: {}'.format(loss))
                #     print('\taccuracy: {}'.format(accuracy))
                step += 1

        except KeyboardInterrupt:
            print('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    main()
