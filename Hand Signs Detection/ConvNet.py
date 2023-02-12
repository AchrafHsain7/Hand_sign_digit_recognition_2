import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import tensorflow as tf
from tensorflow import keras
import keras.layers as tfl
from tensorflow.python.ops.numpy_ops import np_config
from PIL import Image
from helpers import convert_to_one_hot
import pandas as pd

np_config.enable_numpy_behavior()


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



def preprocess_data(X_train, Y_train, X_test, Y_test):
    X_train = X_train/255
    X_test = X_test/255
    Y_train = convert_to_one_hot(Y_train, 6).T
    Y_test = convert_to_one_hot(Y_test, 6).T

    return X_train, Y_train, X_test, Y_test


def convolutional_model(input_shape, hparameters=None):

    #instantiating a keras tensor
    input_img = tf.keras.Input(shape=input_shape)

    #creating the first convolutional layer 
    Z1 = tfl.Conv2D(filters=200, kernel_size=(4,4), strides=2, padding='same')(input_img)
    #ReLu activation function
    A1 = tfl.ReLU()(Z1)
    #first poolong layer
    P1 = tfl.MaxPool2D(pool_size=(4,4), strides=4, padding='same')(A1)
    #second conv layer
    Z2 = tfl.Conv2D(filters=100, kernel_size=(2,2), strides=1, padding='same')(P1)
    #ReLu
    A2 = tfl.ReLU()(Z2)
    #scong max pool layer
    P2 = tfl.MaxPool2D(pool_size=(2,2), strides=2, padding='same')(A2)
    #flattening the data
    F = tfl.Flatten()(P2)
    #creating a fully coneceted layer with softmax activation
    output = tfl.Dense(units=6, activation='softmax')(F)

    model = tf.keras.Model(inputs=input_img, outputs=output)

    return model

#creating and compiling the model
def create_model():
    conv_model = convolutional_model((64,64,3))    
    conv_model.compile(optimizer='adam',
                        loss = 'categorical_crossentropy',
                        metrics=['accuracy'])

    return conv_model


def train():
    #training the model
    X_train, Y_train, X_test, Y_test, classes = load_dataset()
    X_train, Y_train, X_test, Y_test = preprocess_data(X_train, Y_train, X_test, Y_test)
    old_conv_model = keras.models.load_model('model.tf')

    conv_model = create_model()

    #transforming data into tensors and batching them 
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
    history = conv_model.fit(train_dataset, epochs=70, validation_data=test_dataset)

    new_metrics = conv_model.evaluate(test_dataset)
    old_metrics= old_conv_model.evaluate(test_dataset)
    
    conv_model.summary()
    if new_metrics[1] > old_metrics[1]:
        conv_model.save('model.tf')
        print('New Model Saved\nOld accuracy: ' + str(old_metrics[1]) +'\nNew accuracy: ' + str(new_metrics[1]))
    else:
        print('New model NOT saved\nOld accuracy: ' + str(old_metrics[1]) +'\nNew accuracy: ' + str(new_metrics[1]))
    

    df_loss_acc = pd.DataFrame(history.history)
    df_loss= df_loss_acc[['loss','val_loss']]
    #df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
    df_acc= df_loss_acc[['accuracy','val_accuracy']]
    #df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
    df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
    df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
    plt.show()

#train()




    

