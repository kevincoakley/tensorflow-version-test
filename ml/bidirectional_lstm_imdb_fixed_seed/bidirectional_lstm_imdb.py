"""
Title: Bidirectional LSTM on IMDB
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/05/03
Last modified: 2022/04/11 (kevincoakley)
Description: Train a 2-layer bidirectional LSTM on the IMDB movie review sentiment classification dataset.
"""
"""
## Setup
"""

import csv, os, random, sys, yaml
from datetime import datetime 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

script_version = "1.2.0"
max_features = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review

def bidirectional_lstm_imdb(run_number):
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
    ## Build the model
    """

    epochs = 2

    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    x = layers.Embedding(max_features, 128)(inputs)
    # Add 2 bidirectional LSTMs
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    # Add a classifier
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.summary()

    """
    ## Load the IMDB movie review sentiment data
    """

    (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
        num_words=max_features
    )
    print(len(x_train), "Training sequences")
    print(len(x_val), "Validation sequences")
    # Use pad_sequence to standardize sequence length:
    # this will truncate sequences longer than 200 words and zero-pad sequences shorter than 200 words.
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

    """
    ## Train and evaluate the model

    You can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/bidirectional-lstm-imdb)
    and try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/bidirectional_lstm_imdb).
    """

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    #model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_split=0.1)
    model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))

    """
    ## Evaluate the trained model
    """
    score = model.evaluate(x_val, y_val, verbose=0)

    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    """
    ## Save the model
    """
    run_name = os.path.basename(sys.argv[0]).split('.')[0]

    if len(sys.argv) >= 2:
      run_name = run_name + "_" + sys.argv[1]

    y_predicted = model.predict(x_val)
    np.save(run_name + "_predict_" + str(run_number) + ".npy", y_predicted)
    model.save(run_name + "_model_" + str(run_number) + ".h5")


    return score[0], score[1], epochs


def get_system_info():
    if os.path.exists("system_info.py"):
        import system_info
        sysinfo = system_info.get_system_info()

        with open("bidirectional_lstm_imdb_system_info.yaml", "w") as system_info_file:
            yaml.dump(sysinfo, system_info_file, default_flow_style=False)

        return sysinfo
    else:
        return None


def save_score(test_loss, test_accuracy, epochs):
    csv_file = os.path.basename(sys.argv[0]).split('.')[0] + ".csv"
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

    system_info = get_system_info()

    for x in range(5):
        print("\nBidirectional LSTM on IMDB Count: %s\n======================\n" % str(x + 1))
        test_loss, test_accuracy, epochs = bidirectional_lstm_imdb(x + 1)
        save_score(test_loss, test_accuracy, epochs)
