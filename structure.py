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
        self.accuracy = self.get_accuracy(self.logits, self.labels)

    def layers(self, X_inputs, training):
        conv1_1 = tf.layers.conv2d(X_inputs, 32, 5, padding='same', activation=tf.nn.relu)
        conv1_2 = tf.layers.conv2d(conv1_1, 32, 3, padding='same', activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(conv1_2, 2, 2)
        dropout1 = tf.layers.dropout(pool1, 0.7, training=training)

        conv2_1 = tf.layers.conv2d(dropout1, 64, 3, padding='same', activation=tf.nn.relu)
        conv2_2 = tf.layers.conv2d(conv2_1, 64, 3, padding='same', activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(conv2_2, 2, 2)
        dropout2 = tf.layers.dropout(pool2, 0.7, training=training)

        conv3_1 = tf.layers.conv2d(dropout2, 128, 3, padding='same', activation=tf.nn.relu)
        conv3_2 = tf.layers.conv2d(conv3_1, 128, 3, padding='same', activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(conv3_2, 2, 2)
        dropout3 = tf.layers.dropout(pool3, 0.7, training=training)

        conv4_1 = tf.layers.conv2d(dropout3, 256, 3, padding='same', activation=tf.nn.relu)
        conv4_2 = tf.layers.conv2d(conv4_1, 256, 3, padding='same', activation=tf.nn.relu)
        conv4_3 = tf.layers.conv2d(conv4_2, 256, 3, padding='same', activation=tf.nn.relu)
        pool4 = tf.layers.max_pooling2d(conv4_3, 2, 2)
        flat4 = tf.contrib.layers.flatten(pool4)

        d1 = tf.layers.dense(flat4, 512)
        d2 = tf.layers.dense(d1, 512)
        out = tf.layers.dense(d2, config.class_num)
        return tf.nn.softmax(out)

    def optimize(self, logits, labels):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        optimizer = tf.train.AdamOptimizer(config.learning_rate, config.beta1).minimize(loss)
        return loss, optimizer

    def get_accuracy(self, logits, labels):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # accuracy = tf.metrics.accuracy(
        #     labels=tf.argmax(labels, axis=1), predictions=tf.argmax(logits, axis=1))[1]
        return accuracy