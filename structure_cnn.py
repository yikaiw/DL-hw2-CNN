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

        self.logits = self.layers(self.X_inputs)
        self.loss, self.optimizer = self.optimize(self.logits, self.labels)
        self.correct_pre_num = self.get_correct_pre_num(self.logits, self.labels)
        
    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def layers(self, X_inputs):
        self.weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([7 * 7 * 64 * 64, 1024])),
            # 1024 inputs, 64 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, config.class_num + 1]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([config.class_num + 1]))
        }
        
        # Convolution Layer
        conv1 = self.conv2d(X_inputs, self.weights['wc1'], self.biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = self.maxpool2d(conv1, k=2)
        self.shapeconv1 = tf.shape(conv1)
        self.XShape = tf.shape(X_inputs)
        print(X_inputs.shape)
    
        # Convolution Layer
        conv2 = self.conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = self.maxpool2d(conv2, k=2)
        self.shapeconv2 = tf.shape(conv2)
    
        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        # fc1 = tf.reshape(conv2, [-1, self.weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.contrib.layers.flatten(conv2)
        # fc1 = tf.reshape(conv2, [FLAGS.batch_size, 7*7*64])
        fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, self.keep_prob)
        self.shapefc1 = tf.shape(fc1)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])
        return tf.nn.softmax(out)

    def optimize(self, logits, labels):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        optimizer = tf.train.AdamOptimizer(config.learning_rate).minimize(loss)
        return loss, optimizer

    def get_correct_pre_num(self, logits, labels):
        correct_pre = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct_pre_num = tf.reduce_sum(tf.cast(correct_pre, tf.float32))
        # accuracy = tf.metrics.accuracy(
        #     labels=tf.argmax(labels, axis=1), predictions=tf.argmax(logits, axis=1))[1]
        return correct_pre_num
