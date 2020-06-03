# -*- coding: utf-8 -*-
"""
Created on Sat May 11 15:14:06 2019

@author: no295d
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

#rbf algorithm 
order = 4
c = .02
forgetting_factor = 0.97

training_size = 101
num_iter = 1000000

raw_data = np.genfromtxt('wing_data.csv', delimiter=',')

raw_data= raw_data[raw_data[:,1]==0]
train_in = np.array([[x,y] for x, y in zip(raw_data[:,0],raw_data[:,2])], dtype=np.float32)
train_out = raw_data[:,-1]

def normalize_train_data(_x):
    #_X1 = np.reshape(_x[:,0], [-1, 1])
    #_X2 = normalize(np.reshape(_x[:,1], [-1, 1]), axis=0)
    #return  np.concatenate((_X1, _X2), axis=1)
    return normalize(_x, axis=1)

normed_train_data = normalize_train_data(train_in)

def weight_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def weight_constant(shape):
    return tf.ones(shape)

def bias_variable(shape):
    initial = tf.constant(.01, shape = shape)
    return tf.Variable(initial)

def Net(_x, _weights, _bias, _centers, _c):
    '''
    computes the output of an rbf layer in NN
    _x =  [[0.1, 0.3],
           [0.3, -0.4],
           ...]    shape = [train_size, sample_size]
    _weights = [[w11, w12, w13],
                [w21, w22, w23],
                ...] shape = [sample_size, order]
    _centers = [c1, c2, c3] shape = [order]
    '''
    _temp = rbf(_x, _centers, _c)  + _bias['b1']
    
    _temp = tf.matmul(_temp, _weights['w_out'])
    
    return tf.reshape(_temp, [-1]) + _bias['b2']


def rbf(_x, _centers, _c):
    #output is sqrt(sum((x-center)^2) + c)
    _temp = tf.map_fn(lambda center : _x - center, _centers)
    _temp = tf.pow(_temp, 2)
    _temp = tf.reduce_sum(_temp, axis=2) + _c
    _temp = tf.sqrt(_temp)
    return tf.transpose(_temp)


weights = {
    'w_out': weight_variable([order,1])
}


biases = {
    'b1': bias_variable([order]),
    'b2': bias_variable([1])
}

def get_centers(_x, _order):
    '''
    generates centers from the given data points,
    randomly selecting 
    '''
    low = _x.min()
    high = _x.max()
    return tf.convert_to_tensor(
            np.random.uniform(low=low, 
                              high=high, 
                              size=(_order,2)), 
            dtype=tf.float32)
    
centers = get_centers(normed_train_data[:,1], order)

output = Net(normed_train_data, weights, biases, centers, c)

error = output - train_out

loss = tf.reduce_sum(tf.pow(error, 2))/(training_size)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)
print(centers.eval(session=sess))
# train
for step in range(num_iter):
    sess.run(train)
    if step % (num_iter/10) == 0:
        print('Prediction Loss: {}'.format(sess.run(loss)))
        # print(sess.run(temp))
print(weights['w_out'].eval(session=sess))
print(centers.eval(session=sess))
print(biases['b1'].eval(session=sess))
print(biases['b2'].eval(session=sess))
pred = output.eval(session=sess)#Net(normed_train_data, weights, biases, centers).eval(session=sess).flatten()

plt.figure()
plt.plot(pred)
plt.plot(train_out)
plt.show()