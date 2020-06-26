import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


# %% variables and data import
order = 4
c = .02
forgetting_factor = 0.97

training_size = 101
num_iter = 10

raw_data = np.genfromtxt('sin.csv', delimiter=',')
raw_data = raw_data[0:training_size]

#raw_data= raw_data[raw_data[:,1]==0]
train_data = np.array([[x,y] for x, y in zip(raw_data[:,0],raw_data[:,1])], dtype=np.float32)

plt.figure()
plt.plot(train_data[:,1])
plt.show()

training_error = []

def normalize_train_data(_x):
    #_X1 = np.reshape(_x[:,0], [-1, 1])
    #_X2 = normalize(np.reshape(_x[:,1], [-1, 1]), axis=0)
    #return  np.concatenate((_X1, _X2), axis=1)
    return normalize(_x, axis=1)

normed_train_data = normalize_train_data(train_data)

train_in = normed_train_data[:,0]
train_out = normed_train_data[:1]

def weight_constant(shape):
    return np.ones(shape) * .01

def bias_constant(shape):
    return np.ones(shape) * .01

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
    _temp = np.matmul(_rbf, _weights['w_out'])
    
    return np.reshape(_temp, [-1])


def rbf(_x, _centers, _c):
    #output is sqrt(sum((x-center)^2) + c)
    _temp = _x - _centers
    _temp = np.pow(_temp, 2)
    _temp = np.reduce_sum(_temp, axis=2) + _c
    _temp = np.sqrt(_temp)
    return np.transpose(_temp)


weights = {
    'w_out': weight_constant([order + 1,1])
}


biases = {
    'b1': bias_constant([training_size, 1]),
    'b2': bias_constant([training_size])
}

def get_centers(_x, _order):
    '''
    generates centers from the given data points,
    randomly selecting points within the range
    of input data 
    '''
    low = _x.min()
    high = _x.max()
    return np.random.uniform(low=low, 
                            high=high, 
                            size=(_order,2))
    
centers = get_centers(normed_train_data[:,1], order)

# %% run     
covariance = np.diag(np.ones(order + 1)) * 1.0e9

#debug set values for centers
#centers = tf.Variable([[.1, .2], [.3, .4],
#                       [.2, .3], [.4, .5]])
# 

print(centers)
# train
for step in range(num_iter):
    
    rbf_out = rbf(train_in, centers, c)
    rbf_out = np.concat([rbf_out, biases['b1']], 1)
    print("RBF output layer:")
    print(rbf_out)
    output = np.matmul(rbf_out, weights['w_out'])
    error = train_out - np.reshape(output, [-1])
    
    #update weights
    for i in range(training_size):
        rbf_out_temp = rbf_out#np.reshape(rbf_out[i], [order+1, 1])
        _temp = np.matmul(covariance, rbf_out_temp)
        
        _denom = np.matmul(np.transpose(_temp), rbf_out_temp) + forgetting_factor
        _denom = np.cond(np.reduce_max(np.abs(_denom)) < 1e-9,
                        lambda: np.cond(np.reduce_max(np.abs(_denom)) < 0.0,
                                       lambda: np.reshape(np.constant(-1e-9), [1, 1]),
                                       lambda: np.reshape(np.constant(1e-9), [1, 1])
                                       ),
                        lambda: _denom
                        )
        g = np.divide(_temp, _denom)
        gx = np.matmul(g, np.transpose(rbf_out_temp))
        gxp = np.matmul(gx, covariance)
        
        covariance = (covariance - gxp) / forgetting_factor
        weights['w_out'] = weights['w_out'] + error[i] * g
        
    print('Prediction Loss: {}'.format(error))
    training_error.append(error)
        # print(sess.run(temp))
print(weights['w_out'])
print(centers)
#print(biases['b1'].eval(session=sess))
#print(biases['b2'].eval(session=sess))
rbf_out = rbf(normed_train_data, centers, c)
rbf_out = np.concat([rbf_out, bias_constant([101, 1])] ,1)
pred = Net(normed_train_data, weights, biases, rbf_out)

plt.figure()
plt.plot(pred)
plt.plot(train_out)
plt.savefig('pred_vs_actual.png')
plt.show()

plt.figure()
plt.plot(training_error)
plt.savefig('training_error.png')
plt.show()
