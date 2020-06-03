# -*- coding: utf-8 -*-
"""
Created on Wed May  8 22:16:41 2019

@author: no295d
"""
# %%
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

column_names = ['t_s','t_l','x','out']
raw_dataset = pd.read_csv('wing_data.csv', names=column_names, 
                          skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()

# train on t_s and x  first
initial_dataset = pd.DataFrame(dataset)[dataset['t_l'] == 0]

initial_dataset.pop('t_l')

train_dataset = initial_dataset.sample(frac=0.8)
test_dataset = initial_dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("out")
train_stats = train_stats.transpose()

train_out = train_dataset.pop('out')
test_out = test_dataset.pop('out')

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# %%


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float64)
    return tf.Variable(initial)

def weight_constant(shape):
    return tf.ones(shape, dtype=tf.float64)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape, dtype=tf.float64 )
    return tf.Variable(initial)

def rbf_layer(_x, _weights, _bias, _centers):
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
    temp = tf.matmul(_x, _weights['w1']) + _bias['b1']
    temp = tf.subtract(temp,tf.reshape([centers] * 81, [-1,4]))
    return tf.matmul(temp, _weights['w2'])

def Net(_x, _weights,  _bias, _centers):
    return rbf_layer(_x, _weights, _bias, _centers)


weights = {
    'w1': weight_constant([2,4]),
    'w2': weight_variable([4,1])
}


biases = {
    'b1': bias_variable([4])
}

def get_centers(_x, _order):
    '''
    generates centers from the given data points,
    randomly selecting 
    '''
    low = tf.reduce_min(_x)#tf.cast(tf.reduce_min(_x), tf.float32)
    high = tf.reduce_max(_x)#tf.cast(tf.reduce_max(_x), tf.float32)
    return tf.random.uniform((_order,), minval=low, maxval=high, dtype=tf.float64)
    
centers = get_centers(normed_train_data['t_s'], 4)

output = Net(normed_train_data, weights, biases, centers)

error = tf.subtract(output, train_out)
loss = tf.reduce_sum(tf.pow(error, 2))/(81)

optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)
# train
for step in range(20000):
    sess.run(train)
    if step % 2000 == 0:
        print('Prediction Loss: {}'.format(sess.run(loss)))
        # print(sess.run(temp))



# %%












def build_model():
    model = keras.Sequential([
    layers.Dense(4, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

model = build_model()

history = model.fit(
        normed_train_data, train_out,
        epochs=1000, validation_split = 0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,.3])
  plt.legend()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,.15])
  plt.legend()
  plt.show()


plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, test_out, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_out, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

output = initial_dataset.pop('out')

plt.figure()
plt.plot(model.predict(norm(initial_dataset)).flatten(), label='Predict')
plt.plot(output, label='Given')
plt.legend()
plt.show()


# %%
