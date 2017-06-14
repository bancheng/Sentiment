#! /usr/bin/env python
#-*- coding: utf-8 -*-import numpy
import tensorflow as tf
import numpy as np
a = tf.truncated_normal(shape=(1,5),stddev=0.001)
sess = tf.Session()
print sess.run([a])