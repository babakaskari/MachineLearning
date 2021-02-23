import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import yaml

import warnings
import seaborn as sns
sns.set()

warnings.filterwarnings('ignore')

# def label_write(plt, x_axis, y_axis):
#         for x,y in zip(x_axis, y_axis):
#             label = "{:.2f}".format(y)
#             plt.annotate(label, # this is the text
#             (x,y), # this is the point to label
#             textcoords="offset points", # how to position the text
#             xytext=(0,10), # distance from text to points (x,y)
#             ha='center') # horizontal alignment can be left, right or center


def train_val_loss_plotter(history):
    with open("initializer.yaml") as stream:
        param = yaml.safe_load(stream)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(param["fit"]["epochs"])
    plt.figure()
    plt.plot(epochs, loss, 'b', c='red', label='Training loss')
    plt.plot(epochs, val_loss, 'b', c='blue', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def precision_recall_plotter(error_df):
    # mse = np.mean(np.power(df_valid_x_rescaled - valid_x_predictions, 2), axis=1)
    # print("error_def : ", error_df)
    precision, recall, threshold = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
    print("error_df : ", error_df)
    print("Precision : ", precision)
    print("Recall : ", recall)
    print("Threshold : ", threshold)
    plt.plot(threshold, precision[1:], label="Precision", linewidth=5)
    plt.plot(threshold, recall[1:], label="Recall", linewidth=5)
    plt.title('Precision and recall for different threshold values')
    plt.xlabel('Threshold')
    plt.ylabel('Precision/Recall')
    plt.legend()
    plt.show()


def confusion_matrix_plotter(threshold_fixed, LABELS, error_df):
    pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
    conf_matrix = confusion_matrix(error_df.True_class, pred_y)
    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()


def test_plotter(threshold_fixed, x_test, test_prediction, df_test):
    mse = np.mean(np.power(x_test - test_prediction, 2), axis=1)
    error_df_test = pd.DataFrame({'Reconstruction_error': mse,
                                  'True_class': df_test})
    error_df_test = error_df_test.reset_index()

    groups = error_df_test.groupby('True_class')
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
                label="Break" if name == 1 else "Normal")
    ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
    ax.legend()
    plt.title("Reconstruction error for different classes")
    plt.ylabel("Reconstruction error")
    plt.xlabel("Data point index")
    plt.show()


def reconstruction_error(data, decoded_imgs, dim):

    plt.plot(data[0], 'b')
    plt.plot(decoded_imgs[0], 'r')
    plt.fill_between(np.arange(dim), decoded_imgs[0], data[0], color='lightcoral')
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.show()


def histogram_plotter(loss, x_label, y_label):
    plt.hist(loss, bins=50)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def wave_plotter(data, chart_title, dim):
    plt.grid()
    plt.plot(np.arange(dim), data[0])
    plt.title(chart_title)
    plt.show()


def auc_roc_curved(error_df):
    false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df.True_class, error_df.Reconstruction_error)
    roc_auc = auc(false_pos_rate, true_pos_rate,)
    # plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)
    # plt.plot([0, 1], [0, 1], linewidth=5)
    plt.plot(false_pos_rate, true_pos_rate, color="orange", label='AUC = %0.3f' % roc_auc)
    plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--", label='Guessing')
    plt.xlim([-0.01, 1])
    plt.ylim([0, 1.01])
    plt.legend(loc='lower right')
    plt.title('Receiver operating characteristic curve (ROC)')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    print("AUC is  : ", roc_auc)
    plt.show()


def lstm_train_val_loss_plotter(history, epoch):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(epoch)
    plt.figure()
    plt.plot(epochs, loss, 'b', c='red', label='Training loss')
    plt.plot(epochs, val_loss, 'b', c='blue', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
