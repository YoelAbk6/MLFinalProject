import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


def save_confusion_matrix(y_test, y_pred, labels, filepath, algo_name):
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
           title=f'{algo_name} Confusion matrix',
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
    plt.clf()


def save_corr_matrix(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Create a correlation matrix
    corr_matrix = data.df.corr()

    # Create a mask for the lower triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Create a heatmap of the correlation matrix
    sns.heatmap(corr_matrix, annot=True,
                cmap='coolwarm', mask=mask, cbar=False)

    plt.show()
    # Save the heatmap to a file
    # plt.savefig(path)
    plt.clf()


def save_outcome_dist(y):
    num_ones = sum(y)
    num_zeros = len(y) - num_ones

    labels = ['Has Diabetes (1)', 'No Diabetes (0)']
    sizes = [num_ones, num_zeros]
    colors = ['#FFA07A', '#ADD8E6']

    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Outcome Distribution')
    plt.show()


def save_features_density(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fig, axs = plt.subplots(4, 2)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    axs = axs.flatten()
    sns.histplot(data.df['Pregnancies'], kde=True,
                 color='#38b000', ax=axs[0], kde_kws=dict(cut=3))
    sns.histplot(data.df['Glucose'], kde=True,
                 color='#FF9933', ax=axs[1], kde_kws=dict(cut=3))
    sns.histplot(data.df['BloodPressure'], kde=True,
                 color='#522500', ax=axs[2], kde_kws=dict(cut=3))
    sns.histplot(data.df['SkinThickness'], kde=True,
                 color='#66b3ff', ax=axs[3], kde_kws=dict(cut=3))
    sns.histplot(data.df['Insulin'], kde=True,
                 color='#FF6699', ax=axs[4], kde_kws=dict(cut=3))
    sns.histplot(data.df['BMI'], color='#e76f51',
                 kde=True, ax=axs[5], kde_kws=dict(cut=3))
    sns.histplot(data.df['DiabetesPedigreeFunction'],
                 color='#03045e', kde=True, ax=axs[6], kde_kws=dict(cut=3))
    sns.histplot(data.df['Age'], kde=True,
                 color='#333533', ax=axs[7], kde_kws=dict(cut=3))
    plt.show()
    # plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.clf()


def save_ROC(algo_name, path, y_test, y_prob):
    # compute ROC curve and AUC score
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Calculate Youden's J statistic for each threshold
    j_scores = tpr - fpr
    # Find the threshold that maximizes the Youden's J statistic
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    # Create the necessary directories if they don't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver operating characteristic - {algo_name}')
    plt.legend(loc="lower right")
    plt.savefig(path)
    plt.clf()

    return best_threshold


def print_percent(y_test, y_pred, algo):
    count = 0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            count += 1

    print("{}: {:.1f}%".format(algo, (count/len(y_test)) * 100))

