from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from tensorflow.keras.datasets import mnist as keras_mnist

def data_mnist(datadir='/tmp/', train_start=0, train_end=60000, test_start=0, test_end=10000):
    """
    Load MNIST dataset using keras.datasets (works with TF2.19+)
    """
    (X_train, Y_train), (X_test, Y_test) = keras_mnist.load_data()

    # Slice ranges
    X_train = X_train[train_start:train_end]
    Y_train = Y_train[train_start:train_end]
    X_test  = X_test[test_start:test_end]
    Y_test  = Y_test[test_start:test_end]

    # Normalize images to [0,1] and add channel dimension
    X_train = np.expand_dims(X_train.astype(np.float32) / 255.0, axis=-1)
    X_test  = np.expand_dims(X_test.astype(np.float32) / 255.0, axis=-1)

    # Convert labels to one-hot encoding
    Y_train = np.eye(10)[Y_train]
    Y_test  = np.eye(10)[Y_test]

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    return X_train, Y_train, X_test, Y_test



