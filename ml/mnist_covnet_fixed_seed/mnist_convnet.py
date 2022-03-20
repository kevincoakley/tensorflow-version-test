"""
Title: Simple MNIST convnet
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2015/06/19
Last modified: 2022/03/19 (kevincoakley)
Description: A simple convnet that achieves ~99% test accuracy on MNIST.
"""

"""
## Setup
"""

import csv, os, random, sys
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

script_version = "1.0.0"

def simple_mnist_covnet():
    """
    ## Configure Tensorflow for reproducible results
    """

    # Fix the random seed for reproducible results with Keras
    seed_val = 1

    if float(tf.version.VERSION[0:3]) >= 2.7:
        # Sets all random seeds for the program (Python, NumPy, and TensorFlow).
        # Supported in TF 2.7.0+
        tf.keras.utils.set_random_seed(seed_val)
    else:
        # for TF < 2.7
        random.seed(seed_val)
        np.random.seed(seed_val)
        tf.random.set_seed(seed_val)

    # Configures TensorFlow ops to run deterministically to enable reproducible 
    # results with GPUs (Supported in TF 2.8.0+)
    if float(tf.version.VERSION[0:3]) >= 2.8:
        tf.config.experimental.enable_op_determinism()

    """
    ## Prepare the data
    """
    
    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)

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
    csv_file = "mnist_convnet.csv"
    run_name = ""
    write_header = False

    if len(sys.argv) >= 2:
      run_name = sys.argv[1]

    if not os.path.isfile(csv_file):
        write_header = True
      
    with open(csv_file, "a") as csvfile:
        fieldnames = ["run_name", "script_version", "date_time", "python_version", "tensorflow_version",
        "tensorflow_compiler_version", "epochs", "test_loss", "test_accuracy"]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow({"run_name": run_name, "script_version": script_version, "date_time": datetime.now(), "python_version": sys.version, "tensorflow_version": tf.version.VERSION,
        "tensorflow_compiler_version": tf.version.COMPILER_VERSION, "epochs": epochs, "test_loss": test_loss, "test_accuracy": test_accuracy})


if __name__ == '__main__':

    for x in range(5):
        print("\nMNIST Covnet Count: %s\n======================\n" % str(x + 1))
        test_loss, test_accuracy, epochs = simple_mnist_covnet()
        save_score(test_loss, test_accuracy, epochs)
