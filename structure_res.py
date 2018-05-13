import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import config
import resnet18


class CNN(object):
    def __init__(self):
        self.X_inputs = tf.placeholder(tf.float32, [None, config.img_size, config.img_size, 3])
        self.y_inputs = tf.placeholder(tf.int32, [None])
        self.training = tf.placeholder(tf.bool)

        self.labels = tf.one_hot(self.y_inputs, config.class_num, axis=1)
        self.logits = self.layers(self.X_inputs, self.training)
        self.loss, self.optimizer = self.optimize(self.logits, self.labels)
        self.correct_pre_num = self.get_correct_pre_num(self.logits, self.labels)

    def layers(self, X_inputs, training):
        network = resnet18
        model_out = network.inference(X_inputs)
        out = slim.fully_connected(model_out, config.class_num, activation_fn=None,
                                   weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                   weights_regularizer=slim.l2_regularizer(config.weight_decay),
                                   scope='Logits', reuse=False)
        return tf.nn.softmax(out)

    def optimize(self, logits, labels):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        optimizer = tf.train.RMSPropOptimizer(config.learning_rate).minimize(loss)
        return loss, optimizer

    def get_correct_pre_num(self, logits, labels):
        correct_pre = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct_pre_num = tf.reduce_sum(tf.cast(correct_pre, tf.float32))
        return correct_pre_num