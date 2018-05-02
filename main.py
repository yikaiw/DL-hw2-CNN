import tensorflow as tf
from structure import CNN
from datetime import datetime
import os
from tfrecord_reader import tfrecord_read

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('dataset', 'dset1', 'choose dset1 or dset2')
tf.flags.DEFINE_string('pretrained', None, 'for continue training')


def main():
    checkpoint_dir = 'checkpoints'
    if FLAGS.pretrained is not None:
        checkpoint_path = os.path.join(checkpoint_dir, FLAGS.pretrained.lstrip('checkpoints/'))
    else:
        current_time = datetime.now().strftime('%Y%m%d-%H%M')
        checkpoint_path = os.path.join(checkpoint_dir, '{}'.format(current_time))
        try:
            os.makedirs(checkpoint_path)
        except os.error:
            print('Unable to make checkpoints direction: %s' % checkpoint_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    graph = tf.Graph()
    with graph.as_default():
        cnn = CNN()
        read_for_train = tfrecord_read(FLAGS.dataset, True, config.train_slice)
        read_for_val = tfrecord_read(FLAGS.dataset, False, config.train_slice)

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoint_path, graph)
        saver = tf.train.Saver()
    
    with tf.Session(graph=graph, config=config) as sess:
        if FLAGS.pretrained is not None:
            checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
            meta_graph_path = checkpoint.model_checkpoint_path + '.meta'
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
            step = int(meta_graph_path.split('-')[2].split('.')[0])
        else:
            sess.run(tf.global_variables_initializer())
            step = 0
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        try:
            while not coord.should_stop():
                X_train_batch, y_train_batch = sess.run([read_for_train.X_batch, read_for_train.y_batch])
                loss, _ = sess.run([cnn.loss, cnn.optimizer],
                    {cnn.X_inputs: X_train_batch, cnn.y_inputs: y_train_batch, cnn.training: True})

                X_val_batch, y_val_batch = sess.run([read_for_val.X_batch, read_for_val.y_batch])
                accuracy = sess.run(cnn.accuracy,
                    {cnn.X_inputs: X_val_batch, cnn.y_inputs: y_val_batch, cnn.training: False})

                # train_writer.add_summary(summary, step)
                # train_writer.flush()

                if step % 100 == 0:
                    print('At step {}:'.format(step))
                    print('\tloss: {}'.format(loss))
                    print('\taccuracy: {}'.format(accuracy))
                if step % 10000 == 0:
                    save_path = saver.save(sess, checkpoint_path + '/model.ckpt', global_step=step)
                    print('Model saved in file: %s' % save_path)
                step += 1
        
        except KeyboardInterrupt:
            print('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            save_path = saver.save(sess, checkpoint_path + '/model.ckpt', global_step=step)
            print('Model saved in file: %s' % save_path)
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    main()
