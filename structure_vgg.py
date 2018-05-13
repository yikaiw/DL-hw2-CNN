import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import config

tf.set_random_seed(1)
np.random.seed(1)


class CNN(object):
    def __init__(self):
        self.X_inputs = tf.placeholder(tf.float32, [None, config.img_size, config.img_size, 3])
        self.y_inputs = tf.placeholder(tf.int32, [None, ])
        self.labels = tf.one_hot(self.y_inputs, config.class_num, axis=1)
        self.training = tf.placeholder(tf.bool)

        self.logits = self.layers(self.X_inputs, self.training)
        self.loss, self.optimizer = self.optimize(self.logits, self.labels)
        self.correct_pre_num = self.get_correct_pre_num(self.logits, self.labels)

    def layers(self, X_inputs, training):
        input = X_inputs
        filters = [64, 128, 256, 512, 512]
        for filter in filters:
            for i in range(2):
                input = tf.layers.conv2d(input, filter, 3, padding='same', activation=None)
                input = tf.layers.batch_normalization(input, training=training)
                input = tf.nn.relu(input)
            input = tf.layers.max_pooling2d(input, 2, 2)

        flat = tf.contrib.layers.flatten(input)

        d1 = tf.layers.dense(flat, 512)
        d2 = tf.layers.dense(d1, 512)
        out = tf.layers.dense(d2, config.class_num)
        return tf.nn.softmax(out)

    def optimize(self, logits, labels):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        optimizer = tf.train.AdamOptimizer(config.learning_rate, config.beta1).minimize(loss)
        return loss, optimizer

    def get_correct_pre_num(self, logits, labels):
        correct_pre = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct_pre_num = tf.reduce_sum(tf.cast(correct_pre, tf.float32))
        # accuracy = tf.metrics.accuracy(
        #     labels=tf.argmax(labels, axis=1), predictions=tf.argmax(logits, axis=1))[1]
        return correct_pre_num
