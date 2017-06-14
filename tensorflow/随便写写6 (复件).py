#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

W = tf.Variable(initial_value=tf.zeros([784,10]))
b = tf.Variable(initial_value=tf.zeros([10]))
x = tf.placeholder(tf.float32,shape=[None,784])
y_ = tf.placeholder(dtype=tf.float32,shape=[None,10])
y = tf.nn.softmax(tf.matmul(x,W)+b)
cross_entroy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),
                reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(cross_entroy)
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
init.run()

for _ in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train,feed_dict={x:batch_xs,y_:batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
