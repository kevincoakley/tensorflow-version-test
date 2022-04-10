"""
Title: Imbalanced classification: credit card fraud detection
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2019/05/28
Last modified: 2022/04/09 (kevincoakley)
Description: Demonstration of how to handle highly imbalanced classification problems.
"""
"""
## Introduction

This example looks at the
[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud/)
dataset to demonstrate how
to train a classification model on data with highly imbalanced classes.
"""

"""
## First, vectorize the CSV data
"""

import csv, os, random, sys, yaml
from datetime import datetime
import numpy as np
import tensorflow as tf

script_version = "1.2.0"


def imbalanced_classification(run_number):
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


    # Get the real data from https://www.kaggle.com/mlg-ulb/creditcardfraud/
    fname = "creditcard.csv"

    all_features = []
    all_targets = []
    with open(fname) as f:
        for i, line in enumerate(f):
            if i == 0:
                print("HEADER:", line.strip())
                continue  # Skip header
            fields = line.strip().split(",")
            all_features.append([float(v.replace('"', "")) for v in fields[:-1]])
            all_targets.append([int(fields[-1].replace('"', ""))])
            if i == 1:
                print("EXAMPLE FEATURES:", all_features[-1])

    features = np.array(all_features, dtype="float32")
    targets = np.array(all_targets, dtype="uint8")
    print("features.shape:", features.shape)
    print("targets.shape:", targets.shape)

    """
    ## Prepare a validation set
    """

    num_val_samples = int(len(features) * 0.2)
    train_features = features[:-num_val_samples]
    train_targets = targets[:-num_val_samples]
    val_features = features[-num_val_samples:]
    val_targets = targets[-num_val_samples:]

    print("Number of training samples:", len(train_features))
    print("Number of validation samples:", len(val_features))

    """
    ## Analyze class imbalance in the targets
    """

    counts = np.bincount(train_targets[:, 0])
    print(
        "Number of positive samples in training data: {} ({:.2f}% of total)".format(
            counts[1], 100 * float(counts[1]) / len(train_targets)
        )
    )

    weight_for_0 = 1.0 / counts[0]
    weight_for_1 = 1.0 / counts[1]

    """
    ## Normalize the data using training set statistics
    """

    mean = np.mean(train_features, axis=0)
    train_features -= mean
    val_features -= mean
    std = np.std(train_features, axis=0)
    train_features /= std
    val_features /= std

    """
    ## Build a binary classification model
    """

    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.layers.Dense(
                256, activation="relu", input_shape=(train_features.shape[-1],)
            ),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.summary()

    """
    ## Train the model with `class_weight` argument
    """

    model.compile(
        optimizer=keras.optimizers.Adam(1e-2), loss="binary_crossentropy", metrics=["accuracy"]
    )

    #callbacks = [keras.callbacks.ModelCheckpoint("fraud_model_at_epoch_{epoch}.h5")]
    class_weight = {0: weight_for_0, 1: weight_for_1}

    epochs = 30

    model.fit(
        train_features,
        train_targets,
        batch_size=2048,
        epochs=epochs,
        verbose=2,
        #callbacks=callbacks,
        validation_data=(val_features, val_targets),
        class_weight=class_weight,
    )

    score = model.evaluate(val_features, val_targets, verbose=0)

    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    """
    ## Save the model
    """
    run_name = os.path.basename(sys.argv[0]).split('.')[0]

    if len(sys.argv) >= 2:
      run_name = run_name + "_" + sys.argv[1]

    features_predicted = model.predict(val_features)
    np.save(run_name + "_predict_" + str(run_number) + ".npy", features_predicted)
    model.save(run_name + "_model_" + str(run_number) + ".h5")

    return score[0], score[1], epochs


def get_system_info():
    if os.path.exists("system_info.py"):
        import system_info
        sysinfo = system_info.get_system_info()

        with open("imbalanced_classification_system_info.yaml", "w") as system_info_file:
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
        print("\nImbalanced Classification Count: %s\n======================\n" % str(x + 1))
        test_loss, test_accuracy, epochs = imbalanced_classification(x + 1)
        save_score(test_loss, test_accuracy, epochs)

"""
## Conclusions

At the end of training, out of 56,961 validation transactions, we are:

- Correctly identifying 66 of them as fraudulent
- Missing 9 fraudulent transactions
- At the cost of incorrectly flagging 441 legitimate transactions

In the real world, one would put an even higher weight on class 1,
so as to reflect that False Negatives are more costly than False Positives.

Next time your credit card gets  declined in an online purchase -- this is why.
"""
