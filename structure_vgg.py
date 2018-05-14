import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import config


class CNN(object):
    def __init__(self):
        self.X_inputs = tf.placeholder(tf.float32, [None, config.img_size, config.img_size, 3])
        self.y_inputs = tf.placeholder(tf.int32, [None])
        self.labels = tf.one_hot(self.y_inputs, config.class_num, axis=1)
        self.training = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(initial_value=0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(config.learning_rate, self.global_step, 2e3, 1e-4)

        self.logits = self.layers(self.X_inputs)
        self.loss, self.optimizer = self.optimize(self.logits, self.labels)
        self.correct_pre_num, self.batch_accuracy = self.get_correct_pre_num(self.logits, self.labels)

    def layers(self, X_inputs):
        layer = X_inputs
        filters = [64, 128, 256, 512]
        for filter in filters:
            layer = tf.layers.conv2d(layer, filter, 3, padding='same', activation=None)
            layer = tf.layers.batch_normalization(layer, training=self.training)
            layer = tf.nn.relu(layer)
            layer = tf.nn.dropout(layer, self.keep_prob)
            layer = tf.layers.max_pooling2d(layer, 2, 2)

        flat = tf.contrib.layers.flatten(layer)

        d1 = tf.layers.dense(flat, 512)
        d2 = tf.layers.dense(d1, 512)
        out = tf.layers.dense(d2, config.class_num)
        return out

    def optimize(self, logits, labels):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        optimizer = tf.train.AdamOptimizer(
            self.learning_rate, config.beta1, config.beta2).minimize(loss, global_step=self.global_step)
        return loss, optimizer

    def get_correct_pre_num(self, logits, labels):
        softmax_logits = tf.nn.softmax(logits)
        correct_pre = tf.equal(tf.argmax(softmax_logits, 1), tf.argmax(labels, 1))
        correct_pre_num = tf.reduce_sum(tf.cast(correct_pre, tf.float32))
        batch_accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))
        return correct_pre_num, batch_accuracy
