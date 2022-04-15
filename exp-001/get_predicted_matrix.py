#!/usr/bin/env python
"""
Title: Get Predicted Matrix From a list of Tensorflow Results
Author: [kevincoakley](https://github.com/kevincoakley)
Date created: 2022/04/15
Last modified: 2022/04/15
"""
import itertools
import numpy as np


def version():
    """
    :return: version of this script
    """
    return "1.0.0"


def get_predicted_matrix(tensorflow_results, loss_type):
    """
    :param: tensorflow_results: list of output from tensorflow model.predict()
    :param: loss_type: "categorical" or "binary"

    :return: unique_different_indexes: list of indexes that differ in the
        tensorflow_results
    :return: predicted_labels: list of lists of predicted labels
    :return: predicted_values: list of lists of confidence of the prodicted labels
    """
    all_different_indexes = []

    #
    # Find the all of the result indexes that are different
    #

    # Get all combinations of 2 TensorFlow results from the tensorflow_results list
    for two_results_lists in list(itertools.combinations(tensorflow_results, 2)):

        # If the lost_type is categorical, then we need to get the greatest the predicted label
        if loss_type == "categorical":
            # Get the index of the max() item in the list
            first = two_results_lists[0].argmax(axis=1)
            second = two_results_lists[1].argmax(axis=1)
        # If the loss_type is binary, then we need round the predicted label to 0 or 1
        elif loss_type == "binary":
            # Round the values to 0 or 1
            first = np.round_(two_results_lists[0])
            second = np.round_(two_results_lists[1])

        two_results_diff = []

        # Loop through the first two_results_lists list and store the indexes of the results
        # that don't match the results in the second two_results_lists list
        for idx, val in enumerate(first):
            if second[idx] != val:
                two_results_diff.append(idx)

        # Keep a running list of all differnt indexes
        all_different_indexes = all_different_indexes + two_results_diff

    # Remove any duplicate different indexes
    unique_different_indexes = np.unique(all_different_indexes).tolist()

    predicted_labels = []
    predicted_values = []

    # Loop through all of the unique_different_indexes and get results label and the value
    # for idx in unique_different_indexes[:100]:
    for idx in unique_different_indexes:
        labels = []
        values = []
        for item in tensorflow_results:
            if loss_type == "categorical":
                # Get the value by selecting highest value in the results list
                values.append(np.amax(item[idx]))
                # Get the label by selecting the index with the highest value in the results list
                labels.append(np.where(item[idx] == np.amax(item[idx]))[0][0])
            elif loss_type == "binary":
                # Get the value from the results list
                values.append(item[idx][0])
                # Get the label by rounding the result to 0 or 1
                labels.append(int(np.round_(item[idx][0])))

        predicted_labels.append(labels)
        predicted_values.append(values)

    return unique_different_indexes, predicted_labels, predicted_values
