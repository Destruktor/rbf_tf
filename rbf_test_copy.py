# -*- coding: utf-8 -*-
"""
Created on Sat May 11 15:14:06 2019

@author: no295d
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# %% variables and data import
order = 4
c = .02
forgetting_factor = 0.97

training_size = 101
num_iter = 10

raw_data = np.genfromtxt('wing_data.csv', delimiter=',')

raw_data= raw_data[raw_data[:,1]==0]
train_in = np.array([[x,y] for x, y in zip(raw_data[:,0],raw_data[:,2])], dtype=np.float32)
train_out = raw_data[:,-1]

training_error = []

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

def bias_constant(shape):
    return tf.constant(.01, shape = shape)

# %% network
def Net(_x, _weights, _bias, _rbf):
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
    _temp = tf.matmul(_rbf, _weights['w_out'])
    
    return tf.reshape(_temp, [-1])


def rbf(_x, _centers, _c):
    #output is sqrt(sum((x-center)^2) + c)
    _temp = tf.map_fn(lambda center : _x - center, _centers)
    _temp = tf.pow(_temp, 2)
    _temp = tf.reduce_sum(_temp, axis=2) + _c
    _temp = tf.sqrt(_temp)
    return tf.transpose(_temp)


weights = {
    'w_out': weight_variable([order + 1,1])
}


biases = {
    'b1': bias_constant([training_size, 1]),
    'b2': bias_variable([101])
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

#insanity
def update_weights(rbf_out, _covariance, _weights, _forgetting_factor):
    _temp = tf.matmul(_covariance, tf.transpose(rbf_out))
    _temp = tf.reshape(_temp, [1, -1])
    _denom = tf.matmul(_temp, tf.transpose(rbf_out)) + _forgetting_factor
    _denom = tf.cond(tf.reduce_max(tf.abs(_denom)) < 1e-9,
                    lambda: tf.cond(tf.reduce_max(tf.abs(_denom)) < 0.0,
                                   lambda: tf.reshape(tf.constant(-1e-9), [1, 1]),
                                   lambda: tf.reshape(tf.constant(1e-9), [1, 1])
                                   ),
                    lambda: _denom
                    )
    g = tf.div(_temp, _denom)
    gx = tf.matmul(tf.transpose(g), rbf_out)
    gxp = tf.matmul(gx, _covariance)
    _covariance.assign((_covariance - gxp) / forgetting_factor)
    
    _weights['w_out'].assign(
                            _weights['w_out'] + 
                            tf.transpose(tf.matmul(tf.reshape(error, [1, 1]), g))
                            )
    
    return _covariance

# %% run     
covariance = tf.Variable(tf.diag(tf.ones(order + 1) * 1.0e9))

#debug set values for centers
#centers = tf.Variable([[.1, .2], [.3, .4],
#                       [.2, .3], [.4, .5]])
#    

init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)

print(centers.eval(session=sess))
# train
for step in range(num_iter):
    x = tf.placeholder("float")
    t_out = tf.placeholder("float")
    feed = {x:normed_train_data, t_out:train_out}
    
    rbf_out = rbf(x, centers, c)
    rbf_out = tf.concat([rbf_out, biases['b1']], 1)
    #print(sess.run(rbf_out))
    output = tf.matmul(rbf_out, weights['w_out'])
    error = t_out - tf.reshape(output, [-1])
    
    #update weights
    for i in range(training_size):
        rbf_out_temp = tf.reshape(rbf_out[i], [5, 1])
        _temp = tf.matmul(covariance, rbf_out_temp)
        
        _denom = tf.matmul(tf.transpose(_temp), rbf_out_temp) + forgetting_factor
        _denom = tf.cond(tf.reduce_max(tf.abs(_denom)) < 1e-9,
                        lambda: tf.cond(tf.reduce_max(tf.abs(_denom)) < 0.0,
                                       lambda: tf.reshape(tf.constant(-1e-9), [1, 1]),
                                       lambda: tf.reshape(tf.constant(1e-9), [1, 1])
                                       ),
                        lambda: _denom
                        )
        g = tf.divide(_temp, _denom)
        gx = tf.matmul(g, tf.transpose(rbf_out_temp))
        gxp = tf.matmul(gx, covariance)
        
        assign_covariance = covariance.assign((covariance - gxp) / forgetting_factor)
        assign_weights = weights['w_out'].assign(weights['w_out'] + error[i] * g)
        
        h = sess.partial_run_setup([rbf_out, error, assign_covariance, assign_weights], [x, t_out])
        
        sess.partial_run(h, rbf_out, feed_dict=feed)
        sess.partial_run(h, error)
        sess.partial_run(h, assign_covariance)
        sess.partial_run(h, assign_weights)
        
        
    #if step % (num_iter/10) == 0:
    v = sess.run(tf.reduce_sum(tf.abs(error)), feed_dict=feed)
    print('Prediction Loss: {}'.format(v))
    training_error.append(v)
        # print(sess.run(temp))
print(weights['w_out'].eval(session=sess))
print(centers.eval(session=sess))
#print(biases['b1'].eval(session=sess))
#print(biases['b2'].eval(session=sess))
rbf_out = rbf(normed_train_data, centers, c)
rbf_out = tf.concat([rbf_out, bias_constant([101, 1])] ,1)
pred = Net(normed_train_data, weights, biases, rbf_out).eval(session=sess).flatten()

plt.figure()
plt.plot(pred)
plt.plot(train_out)
plt.savefig('training_error.png')
plt.show()