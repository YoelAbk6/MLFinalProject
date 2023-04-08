import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def save_confusion_matrix(y_test, y_pred, labels, filepath):
    """
    Create a confusion matrix from y_test and y_pred, and save it as an image.

    Args:
    - y_test (array-like): True labels of the test data
    - y_pred (array-like): Predicted labels of the test data
    - labels (array-like): List of class labels in order of appearance in confusion matrix
    - filepath (str): Name of the file to save the image as

    Returns:
    - None
    """
    # Create the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot the confusion matrix as an image
    fig, ax = plt.subplots()
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)

    # Add colorbar and labels
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title='Confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm_norm.max() / 2.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, format(cm_norm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black")

    # Create the necessary directories if they don't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save the figure
    fig.tight_layout()
    plt.savefig(filepath)
