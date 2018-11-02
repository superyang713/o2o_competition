import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score


def load_data(filename):
    """
    Load the data into a DataFrame. The data file is located in ./data/
    """
    directory = os.path.abspath(os.path.dirname(__file__))
    filepath = os.path.join(directory, 'data', filename)
    data = pd.read_csv(filepath)

    return data


def save_fig(fig_id, tight_layout=True, fig_extension='png', resolution=300):
    directory = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'image'
    )
    if not os.path.isdir(directory):
        os.mkdir(directory)
    fig_path = os.path.join(directory, fig_id + '.' + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(fig_path, format=fig_extension, dpi=resolution)


def create_submission_file(model, pipeline, filename='submit.csv'):
    test = load_data('ccf_offline_stage1_test_revised.csv')
    X = test.copy()
    X['Date_received'] = pd.to_datetime(
        test['Date_received'], format='%Y%m%d'
    )
    y_test_pred = model.predict_proba(pipeline.transform(X))
    test_copy = test[['User_id', 'Coupon_id', 'Date_received']].copy()
    test_copy['Probability'] = y_test_pred[:, 1]
    test_copy.to_csv(filename, index=False, header=False)
    print('The submission file has been successfully saved. File name: {}'
          .format(filename))


def plot_precision_recall_vs_threshold(ax, precisions, recalls, thresholds):
    ax.plot(thresholds, precisions[:-1], "b--", label='Precision')
    ax.plot(thresholds, recalls[:-1], "g-", label="Recall")
    ax.set_title("Precision vs. Recall")
    ax.set_xlabel("Threshold")
    ax.set_ylim([0, 1])
    ax.legend(loc="center left")


def plot_roc_curve(ax, fpr, tpr):
    ax.plot(fpr, tpr, linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title("ROC Curve")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')


def evaluate_model(model, X_train, y_train, X_val, y_val, plot_train=True):
    try:
        y_train_scores = model.decision_function(X_train)
        y_val_scores = model.decision_function(X_val)
    except AttributeError:
        y_train_scores = model.predict_proba(X_train)[:, 1]
        y_val_scores = model.predict_proba(X_val)[:, 1]

    precisions_train, recalls_train, thresholds_train = \
        precision_recall_curve(y_train, y_train_scores)
    precisions_val, recalls_val, thresholds_val = \
        precision_recall_curve(y_val, y_val_scores)
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_scores)
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_scores)
    roc_train = roc_auc_score(y_train, y_train_scores)
    roc_val = roc_auc_score(y_val, y_val_scores)

    if plot_train:
        f_train, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
        f_train.suptitle("Evaluation on Training Dataset")

        plot_precision_recall_vs_threshold(
            ax1, precisions_train, recalls_train, thresholds_train
        )
        plot_roc_curve(ax2, fpr_train, tpr_train)
        ax2.text(0.5, 0.3, "roc = {}".format(round(roc_train, 3)))

    f_val, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
    f_val.suptitle("Evaluation on Validation Dataset")

    plot_precision_recall_vs_threshold(
       ax1, precisions_val, recalls_val, thresholds_val
    )
    plot_roc_curve(ax2, fpr_val, tpr_val)
    ax2.text(0.5, 0.3, "roc = {}".format(round(roc_val, 3)))
