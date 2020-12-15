import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation
from sklearn import preprocessing, metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder
import sklearn
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from gaussrank import *
from sklearn.metrics import plot_confusion_matrix
from xgboost import XGBClassifier
import prepossessed_dataset

pd.set_option('mode.chained_assignment', None)
dataset = prepossessed_dataset.semi_super()
x_unlabeled = dataset["x_unlabeled"]
x_train = dataset["x_train"]
y_train = dataset["y_train"]
x_test = dataset["x_test"]
y_test = dataset["y_test"]


def evaluate_preds(model, x_true, y_true, y_preds):
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

    return metric_dict
# Initiate iteration counter


iterations = 0

# Containers to hold f1_scores and # of pseudo-labels
train_f1s = []
test_f1s = []
pseudo_labels = []

# Assign value to initiate while loop
high_prob = [1]

# Loop will run until there are no more high-probability pseudo-labels
while len(high_prob) > 0 and len(x_unlabeled) > 0:
    # Fit classifier and make train/test predictions
    # print(y_train)
    # clf = LogisticRegression(max_iter=10000)
    # clf.fit(X_train, y_train.values.ravel())

    # #######################################  XGBClassifier()
    clf = clf_rfc = RandomForestClassifier()
    clf.fit(x_train, y_train)
    # xgb_pred = clf.predict(x_train)
    # xgb_matrices = evaluate_preds(clf, x_train, y_test, xgb_pred)

    y_hat_train = clf.predict(x_train)
    y_hat_test = clf.predict(x_test)
    # Calculate and print iteration # and f1 scores, and store f1 scores
    train_f1 = f1_score(y_train, y_hat_train, average='micro')
    test_f1 = f1_score(y_test, y_hat_test, average='micro')
    print(f"Iteration {iterations}")
    print(f"Train f1: {train_f1}")
    print(f"Test f1: {test_f1}")
    train_f1s.append(train_f1)
    test_f1s.append(test_f1)

    # Generate predictions and probabilities for unlabeled data
    print(f"Now predicting labels for unlabeled data...")

    pred_probs = clf.predict_proba(x_unlabeled)
    preds = clf.predict(x_unlabeled)
    prob_0 = pred_probs[:, 0]
    prob_1 = pred_probs[:, 1]
    # Store predictions and probabilities in dataframe
    df_pred_prob = pd.DataFrame([])
    df_pred_prob['Leak Found'] = preds
    df_pred_prob['prob_0'] = prob_0
    df_pred_prob['prob_1'] = prob_1
    df_pred_prob.index = x_unlabeled.index
    # Separate predictions with > 99% probability
    high_prob = pd.concat([df_pred_prob.loc[df_pred_prob['prob_0'] > 0.99],
                           df_pred_prob.loc[df_pred_prob['prob_1'] > 0.99]],
                          axis=0)
    # print(high_prob)
    print(f"{len(high_prob)} high-probability predictions added to training data.")

    pseudo_labels.append(len(high_prob))

    # Add pseudo-labeled data to training data
    x_train = pd.concat([x_train, x_unlabeled.loc[high_prob.index]], axis=0)
    high_prob = high_prob.drop(columns=['prob_0', 'prob_1'])
    print(high_prob)

    y_train = pd.concat([y_train, high_prob])

    # Drop pseudo-labeled instances from unlabeled data
    x_unlabeled = x_unlabeled.drop(index=high_prob.index)
    print(f"{len(x_unlabeled)} unlabeled instances remaining.\n")

    # Update iteration counter
    iterations += 1
    print(f"Test f1: {test_f1s}")




