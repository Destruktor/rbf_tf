# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:35:29 2019

@author: no295d
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

def rbf(x):
    return tf.norm(x - tf.constant(4.0))
np_rbf = np.vectorize(rbf)

def d_rbf(x):
    return np.divide(x - 1.0, np.linalg.norm(x - 4.0))
np_d_rbf = np.vectorize(d_rbf)

#make 
np_d_rbf_32 = lambda x: np_d_rbf(x).astype(np.float32)
def tf_d_rbf(x,name=None):
    with tf.name_scope(name, "d_rbf", [x]) as name:
        y = tf.py_func(np_d_rbf_32,
                        [x],
                        [tf.float32],
                        name=name,
                        stateful=False)
        return y[0]



if __name__ == '__main__':
    
    x = np.array([1., 8.])
    center = 4.
    
    #assert isinstance(rbf(x), float)
    #assert rbf(x) == 5.0
    
    #assert isinstance(d_rbf(x), np.ndarray)
    
    #define activation function
    
    np_rbf_32 = lambda x: np_rbf(x).astype(np.float32)
    def tf_rbf(x,name=None):
        with tf.name_scope(name, "rbf", [x]) as name:
            y = tf.py_func(np_rbf_32,
                            [x],
                            [tf.float32],
                            name=name,
                            gradient=np_d_rbf)
            return y[0]


    
    
    with tf.Session() as sess:

        x = tf.constant([1., 8.])
        centers = tf.constant([4.])
        y = rbf(x)
        tf.initialize_all_variables().run()
    
        print(x.eval(), y.eval(), tf.gradients(y, [x])[0].eval())