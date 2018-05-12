import numpy as np
import tensorflow as tf
from structure import CNN
from datetime import datetime
import os
from tfrecord_reader import tfrecord_read
import config
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('dataset', 'dset1', 'Choose dset1 or dset2 for training, default dset1.')
tf.flags.DEFINE_string('checkpoint', None, 
                       'Whether use a pre-trained checkpoint to continue training, default None.')


def main():
    checkpoint_dir = 'checkpoints'
    if FLAGS.checkpoint is not None:
        checkpoint_path = os.path.join(checkpoint_dir, FLAGS.checkpoint.lstrip('checkpoints/'))
    else:
        current_time = datetime.now().strftime('%Y%m%d-%H%M')
        checkpoint_path = os.path.join(checkpoint_dir, '{}'.format(current_time))
        try:
            os.makedirs(checkpoint_path)
        except os.error:
            print('Unable to make checkpoints direction: %s' % checkpoint_path)
    model_save_path = os.path.join(checkpoint_path, 'model.ckpt')

    cnn = CNN()
    read_for_train = tfrecord_read(
        FLAGS.dataset, config.batch_size, config.num_epochs, config.train_slice, training=True)
    read_for_val = tfrecord_read(
        FLAGS.dataset, config.batch_size, config.num_epochs, config.train_slice, training=False)

    saver = tf.train.Saver()

    print('Build session.')
    sess = tf.Session()

    if FLAGS.checkpoint is not None:
        print('Restore from pre-trained model.')
        checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
        meta_graph_path = checkpoint.model_checkpoint_path + '.meta'
        restore = tf.train.import_meta_graph(meta_graph_path)
        restore.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
        step = int(meta_graph_path.split('-')[2].split('.')[0])
    else:
        print('Initialize.')
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        step = 0

    epoch_pre = step * config.batch_size // config.file_num[FLAGS.dataset]
    accuracies = []

    # train_writer = tf.summary.FileWriter('log', sess.graph)
    # summary_op = tf.summary.merge_all()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        print('Start training:')
        while not coord.should_stop():
            X_train_batch, y_train_batch = sess.run([read_for_train.X_batch, read_for_train.y_batch])
            loss, _ = sess.run([cnn.loss, cnn.optimizer],
                {cnn.X_inputs: X_train_batch, cnn.y_inputs: y_train_batch, cnn.training: True})

            X_val_batch, y_val_batch = sess.run([read_for_val.X_batch, read_for_val.y_batch])
            accuracy = sess.run(cnn.accuracy,
                {cnn.X_inputs: X_val_batch, cnn.y_inputs: y_val_batch, cnn.training: False})
            accuracies.append(accuracy)

            # train_writer.add_summary(summary, step)
            # train_writer.flush()

            epoch_cur = step * config.batch_size // config.file_num[FLAGS.dataset]
            if epoch_cur > epoch_pre:
                print('For epoch {}: accuracy = {}'.format(epoch_pre, np.mean(accuracies)))
                epoch_pre = epoch_cur
                accuracies = []

            if step % 100 == 0:
                print('At step {}: loss = {}'.format(step, loss))
            if step % 10000 == 0 and step > 0:
                save_path = saver.save(sess, model_save_path, global_step=step)
                print('Model saved in file: %s' % save_path)
            step += 1

    except KeyboardInterrupt:
        print('Interrupted')
        coord.request_stop()
    except Exception as e:
        coord.request_stop(e)
    finally:
        save_path = saver.save(sess, model_save_path, global_step=step)
        print('Model saved in file: %s' % save_path)
        coord.request_stop()
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    main()
