"""
Title: Simple MNIST convnet
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2015/06/19
Last modified: 2022/03/15 (kevincoakley)
Description: A simple convnet that achieves ~99% test accuracy on MNIST.
"""

"""
## Setup
"""

import sys
from datetime import datetime
import numpy as np
from numpy.random import seed
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def simple_mnist_covnet():
    """
    ## Prepare the data
    """

    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)

    # Fix the random seed for reproducible results with Keras
    seed_val = 1
    seed(seed_val)
    tf.random.set_seed(seed_val)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")


    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    """
    ## Build the model
    """

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()

    """
    ## Train the model
    """

    batch_size = 128
    epochs = 15

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    """
    ## Evaluate the trained model
    """

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    
    return score[0], score[1], epochs


def save_score(test_loss, test_accuracy, epochs):
    output_file = "fixed_scores.txt"

    if len(sys.argv) >= 2:
      output_file = "fixed_scores_%s.txt" % (sys.argv[1])

    f = open(output_file, 'a')
    f.write("======================\n")
    f.write("%s\n" % datetime.now())
    f.write("Python: %s\n" % sys.version)
    f.write("TensorFlow version: %s\n" % tf.version.VERSION)
    f.write("TensorFlow compiler version: %s\n" % tf.version.COMPILER_VERSION)
    f.write("Epochs: %d \n" % epochs)
    f.write("Test loss: %0.16f\n" % (test_loss))
    f.write("Test accuracy: %0.16f\n" % (test_accuracy))
    f.close()


if __name__ == '__main__':

    # Configures TensorFlow ops to run deterministically to enable reproducible 
    # results with GPUs (Supported in TF 2.8.0+)
    if float(tf.version.VERSION[0:3]) >= 2.8:
        tf.config.experimental.enable_op_determinism()

    for x in range(5):
        print("\nMNIST Covnet Count: %s\n======================\n" % str(x + 1))
        test_loss, test_accuracy, epochs = simple_mnist_covnet()
        save_score(test_loss, test_accuracy, epochs)
