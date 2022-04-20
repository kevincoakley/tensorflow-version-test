#!/usr/bin/env python
"""
Title: 
Author: [kevincoakley](https://github.com/kevincoakley)
Date created: 2022/04/15
Last modified: 2022/04/15
"""
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def version():
    """
    :return: version of this script
    """
    return "2.0.0"


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


def create_heat_map_from_predicted_matrix(
    predicted_labels, predicted_values, title, xlabel, ylabel, figsize=(10, 10)
):
    """
    :param: predicted_labels: dataframe of predicted labels from get_predicted_matrix
    :param: predicted_values: dataframe of confidence values from get_predicted_matrix
    :param: title: title of the heat map
    :param: xlabel: x-axis label of the heat map
    :param: ylabel: y-axis label of the heat map
    :param: figsize: size of the heat map

    :return: heat_map: heat map of the predicted labels
    """
    sns.set(rc={"figure.figsize": figsize})
    sns.set(font_scale=1.2)

    ax = sns.heatmap(predicted_labels, annot=True, fmt="d", cbar=False, cmap="Blues")

    # Create a flat list of predicted values by using the index of the
    # predicted_labels dataframe to get the value from the
    # predicted_values dataframe
    predicted_values_flat = []
    for idx in predicted_labels.index:
        predicted_values_flat += list(predicted_values.loc[idx])

    # Loop through all of the cell values and add the annotatations to the text
    for idx, val in enumerate(ax.texts):
        val.set_text(val.get_text() + " (" + str(predicted_values_flat[idx]) + ")")

    ax.set(title=title)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)

    return ax


def create_prediction_data_frame_for_index(prediction_index, tensorflow_results, results_names):
    """
    :param: prediction_index: list of indexes that differ in the tensorflow_results
    :param: tensorflow_results: list of confidence values from tensorflow model.predict()
    :param: results_names: list of names for the tensorflow_results

    :return: dataframe of the confidence values for each predicted label
    """
    results_name = []
    label = []
    predicted_value = []

    # Loop through the TensorFlow results from the tensorflow_results list
    for tf_index, tensorflow_result in enumerate(tensorflow_results):

        # For each tensorflow_result get the TensorFlow results name, the label and the predicted value
        for index, value in enumerate(tensorflow_result[prediction_index]):
            results_name.append(results_names[tf_index])
            label.append(index)
            predicted_value.append(value)

    data = {"Name": results_name, "Label": label, "Predicted Value": predicted_value}

    return pd.DataFrame(data)


def create_prediction_barplot_for_index(df, unique_index, col_wrap=3, figsize=(10, 10)):
    """
    :param: df: dataframe of the confidence values and predicted label from create_prediction_data_frame_for_index
    :param: unique_index: list of indexes that differ in the tensorflow_results
    :param: col_wrap: number of columns to wrap the FacetGrid
    :param: figsize: size of the barplot

    :return: FacetGrid of barplots of the confidence values for each predicted label
    """
    sns.set(rc={"figure.figsize": figsize})
   
    fg = sns.FacetGrid(df, col="Name", col_wrap=col_wrap)
    fg.map_dataframe(sns.barplot, y="Label", x="Predicted Value", orient="h")

    fg.fig.suptitle("Test Index %s" % unique_index)

    # Function to add the text to each bar of the barplot
    def annotate(data, **kws):
        # Get Current Axis (the current facet)
        ax = plt.gca()

        # label each bar in barplot
        for p in ax.patches:
            height = p.get_height()
            width = p.get_width()

            # adding text to each bar
            ax.text(
                x=width,
                y=p.get_y() + (height / 2),
                s="{:.2e}".format(
                    float(width)
                ),  # data label, formatted using scientific notation with 2 decimals
                va="center",
            )

    fg.set(xlim=(0, 1.1))
    fg.map_dataframe(annotate)
