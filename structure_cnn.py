import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import config

tf.set_random_seed(1)
np.random.seed(1)


class CNN(object):
    def __init__(self):
        self.X_inputs = tf.placeholder(tf.float32, [None, config.img_size, config.img_size, 3])
        self.y_inputs = tf.placeholder(tf.int32, [None])
        self.labels = tf.one_hot(self.y_inputs, config.class_num, axis=1)
        self.training = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(initial_value=0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(config.learning_rate, self.global_step, 2e3, 1e-4)

        self.logits = self.layers(self.X_inputs)  # without softmax
        self.loss, self.optimizer = self.optimize(self.logits, self.labels)
        self.correct_pre_num, self.batch_accuracy = self.get_correct_pre_num(self.logits, self.labels)
        
    def conv2d(self, x, W, b, strides=1):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def layers(self, X_inputs):
        self.weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            'wd1': tf.Variable(tf.random_normal([7 * 7 * 64 * 64, 512])),
            'out': tf.Variable(tf.random_normal([512, config.class_num]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([512])),
            'out': tf.Variable(tf.random_normal([config.class_num]))
        }
        
        conv1 = self.conv2d(X_inputs, self.weights['wc1'], self.biases['bc1'])
        conv1 = self.maxpool2d(conv1, k=2)
        conv2 = self.conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
        conv2 = self.maxpool2d(conv2, k=2)
        fc1 = tf.contrib.layers.flatten(conv2)
        fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, self.keep_prob)
        self.shapefc1 = tf.shape(fc1)
        out = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])
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
