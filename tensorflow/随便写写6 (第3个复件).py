#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
a = mnist.train.next_batch(200)
b = mnist.train.next_batch(100)

print len(a[0][0])
print "batch_ys"
# print batch_ys