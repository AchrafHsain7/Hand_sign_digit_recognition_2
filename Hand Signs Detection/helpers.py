import numpy as np
import tensorflow as tf
import h5py

def reLU(Z):
        A = np.maximum(0, Z)
        
        return A
    
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))

    return A

def deriv_reLU(dA, activation_cache):
    Z, _ = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[ Z< 0] = 0

    return dZ

def deriv_sigmoid(dA,activation_cache):
    Z,A = activation_cache
    dZ = dA * A*(1-A)

    return dZ

'''''
def convert_to_one_hot(labels, depth):
    #given labels and number of neurons create a one hot version of Y
    one_hot = tf.one_hot(labels, depth, axis=0)
    one_hot = tf.reshape(one_hot , shape=[-1,])


    return one_hot
'''
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def load_dataset():

    h5_train = h5py.File('train_signs.h5', 'r')
    h5_test = h5py.File('test_signs.h5', 'r')

    X_train = np.array(h5_train.get('train_set_x'))
    Y_train = np.array(h5_train.get('train_set_y'))
    X_test = np.array(h5_test.get('test_set_x'))
    Y_test = np.array(h5_test.get('test_set_y'))
    classes = np.array(h5_train.get('list_classes'))

    Y_train = Y_train.reshape((1, Y_train.shape[0]))
    Y_test = Y_test.reshape((1, Y_test.shape[0]))

    return X_train, Y_train, X_test, Y_test, classes
