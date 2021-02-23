
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import sklearn
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# tf.keras.losses.MeanSquaredError(
#     reduction=losses_utils.ReductionV2.AUTO, name='mean_squared_error'
# )
import warnings

# import data_picker
import seaborn as sns
sns.set()

warnings.filterwarnings('ignore')


def evaluate_ann(history):
    # print("Cosine Proximity  :   ", history.history['cosine'])
    print("Train Loss : ", history.history["loss"])
    print("Validation Loss : ", history.history["val_loss"])
    print("Mean Squared Erro : ", history.history['mean_squared_error'])
    print("Mean Absolute Error : ", history.history['mean_absolute_error'])
    print("Mean Absolute Percentage Error : ", history.history['mean_absolute_percentage_error'])
    print("Val Mean Squared Erro : ", history.history['val_mean_squared_error'])
    print("Val Mean Absolute Error : ", history.history['val_mean_absolute_error'])
    print("Val Mean Absolute Percentage Error : ", history.history['val_mean_absolute_percentage_error'])
    print("Train Loss : ", history.history["loss"][0])
    print("Validation Loss : ", history.history["val_loss"][0])
    # plot metrics


def predict(model, data, thresholds):
    reconstruction = model(data)
    loss = tf.keras.losses.mae(reconstruction, data)
    # print("loss :", loss)
    # return tf.math.less(loss, threshold)
    return tf.math.less(thresholds, loss)


def print_stats(predictions, labels):
    print("Accuracy = {}".format(accuracy_score(labels, predictions)))
    print("Precision = {}".format(precision_score(labels, predictions)))
    print("Recall = {}".format(recall_score(labels, predictions)))


def prediction(autoencoder, x_test, y_test, threshold):
    preds = predict(autoencoder, x_test, threshold)
    # print("y_test : ", y_test)
    # print("preds : ", preds)
    print_stats(preds, y_test)
    print(preds)
    confusion_matrix_value = confusion_matrix(y_test, preds)
    print("Confusion Matrix : \n ", confusion_matrix_value)
    print("roc_auc_score : ", roc_auc_score(y_test, preds))
    fpr, tpr, thresholds = roc_curve(y_test, preds)
    # plt.plot(fpr, tpr, color="orange", label="ROC")
    # plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--", label="Guessing")
    plt.plot(fpr, tpr, color="orange", label='AUC = %0.3f' % roc_auc_score(y_test, preds))
    plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--", label='Guessing')
    plt.xlabel("False positive rate (fpr)")
    plt.ylabel("True positive rate (tpr)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.show()







