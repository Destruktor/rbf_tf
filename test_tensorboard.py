# -*- coding: utf-8 -*-
"""
Created on Fri May 17 08:52:30 2019

@author: no295d
"""

import tensorflow as tf
tf.Variable(42, name='foo')
w = tf.summary.FileWriter("C:/Users/no295d/Documents/tensorboard/test")
w.add_graph(tf.get_default_graph())
w.flush()
w.close()