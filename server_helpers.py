import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

def unflatten(arr, rows, cols):
    if len(arr) != rows * cols:
        raise ValueError("The length of the array does not match the specified dimensions.")
    
    arr = np.reshape(arr, (-1, cols))
    # result = []
    # for i in range(rows):
    #     row = arr[i * cols:(i + 1) * cols]
    #     result.append(row)
    # result = np.array(result)
    return arr


# new_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
#                             loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#                             metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
#                         )
def make_prediction(X):
    print('IN MAKE_PREDICTION()')
    new_model = tf.keras.models.load_model('mnist_model.keras')
    X = np.array(X, dtype='float32')
    print(X, '##')
    X = np.reshape(X, (1, 28, 28))
    print(X, '##RESHAPED')
    pred = np.argmax(new_model.predict(X))
    print(pred)
    print(new_model.summary())
    loss, acc = new_model.evaluate(X_test, y_test)
    print(loss, acc)
    
    return pred
