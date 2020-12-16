import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import OneHotEncoder
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import preprocessing, metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from gaussrank import *
import warnings
import seaborn as sns
sns.set()

warnings.filterwarnings('ignore')


def evaluate_preds(model, x_train, y_train,  x_true, y_true, x_cv, y_cv):
    y_preds = model.predict(x_true)
    print("Prediction : ", y_preds)
    accuracy = metrics.accuracy_score(y_true, y_preds)
    precision = metrics.precision_score(y_true, y_preds)
    recall = metrics.recall_score(y_true, y_preds)
    f1 = metrics.f1_score(y_true, y_preds)
    metric_dict = {"accuracy": round(accuracy, 2),
                   "precision": round(precision, 2),
                   "recall": round(recall, 2),
                   "f1": round(f1, 2)}
    print("Model score is : ", model.score(x_true, y_true))
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision : {precision: .2f}")
    print(f"Recall : {recall: .2f}")
    print(f"F1 Score : {f1: .2f}")
    print("Score:", metrics.accuracy_score(y_true, y_preds))
    print("Model score is : ", model.score(x_true, y_true))
    # print(np.mean(y_test == y_pred))
    pred_probality = model.predict_proba(x_true)
    # print("Predict probability : ", pred_probality)
    # cross_validation score
    cross_validation_score = cross_val_score(model, x_train, y_train, cv=6)
    print("Cross validation score : ", cross_validation_score)
    cross_validation_predict = cross_val_predict(model, x_train, y_train, cv=6)
    print("Cross validation predict : ", cross_validation_predict)
    cross_val_accuracy = np.mean(cross_validation_score) * 100
    print("cross validation accuracy : ", cross_val_accuracy)
    # ROC
    print("pred_probality : ", pred_probality, "length of prediction prob : ", len(pred_probality))
    y_probs_positive = pred_probality[:, 1]
    print("y_probs_positive : ", y_probs_positive)
    fpr, tpr, thresholds = roc_curve(y_true, y_probs_positive)
    print("fpr : ", fpr)
    print("roc_auc_score : ", roc_auc_score(y_true, y_probs_positive))
    plt.plot(fpr, tpr, color="orange", label="ROC")
    plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--", label="Gussing")
    plt.xlabel("False positive rate (fpr)")
    plt.ylabel("True positive rate (tpr)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.show()





