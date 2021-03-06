import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import config

tf.set_random_seed(1)
np.random.seed(1)
total_layers = 25 #Specify how deep we want our network
units_between_stride = total_layers // 5

class CNN(object):
    def __init__(self):
        self.X_inputs = tf.placeholder(tf.float32, [None, config.img_size, config.img_size, 3])
        self.y_inputs = tf.placeholder(tf.int32, [None])
        self.labels = tf.one_hot(self.y_inputs, config.class_num, axis=1)
        self.training = tf.placeholder(tf.bool)

        self.logits = self.layers(self.X_inputs, self.training)
        self.loss, self.optimizer = self.optimize(self.logits, self.labels)
        self.correct_pre_num = self.get_correct_pre_num(self.logits, self.labels)

    def unit(self, input_layer, i):
        with tf.variable_scope('unit' + str(i)):
            part1 = slim.batch_norm(input_layer, activation_fn=None)
            part2 = tf.nn.relu(part1)
            part3 = slim.conv2d(part2, 64, [3, 3], activation_fn=None)
            #part4 = slim.batch_norm(part3, activation_fn=None)
            #part5 = tf.nn.relu(part4)
            #part6 = slim.conv2d(part5, 64, [3, 3], activation_fn=None)
            output = input_layer + part3
            return output

    def layers(self, X_inputs, training):
        layer = slim.conv2d(X_inputs, 64, [3, 3], normalizer_fn=slim.batch_norm, scope='conv_' + str(0))
        for i in range(5):
            for j in range(units_between_stride):
                layer = self.unit(layer, j + (i * units_between_stride))
            layer = slim.conv2d(layer, 64, [3, 3], stride=[2, 2], 
                                normalizer_fn=slim.batch_norm, scope='conv_s_' + str(i))
            
        flat = tf.contrib.layers.flatten(layer)
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