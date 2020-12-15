import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import dates as mpl_dates
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.semi_supervised import LabelPropagation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
import prepossessed_dataset
from sklearn.metrics import roc_curve
import seaborn as sns
from sklearn import metrics
sns.set()

dataset = prepossessed_dataset.just_labeled()
print(dataset)
print("x_train  :  ", dataset["x_train"])
print("y_train  :  ", dataset["y_train"])
x_train = dataset["x_train"]
y_train = dataset["y_train"]
x_test = dataset["x_test"]
y_test = dataset["y_test"]
x_cv = dataset["x_cv"]
y_cv = dataset["y_cv"]
print('Number of data points in train data:', x_train.shape[0])
print('Number of data points in test data:', x_test.shape[0])
print('Number of data points in test data:', x_cv.shape[0])
clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=20, random_state=42, n_jobs=-1)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("Prediction : ", y_pred)
print("Score:", metrics.accuracy_score(y_test, y_pred))
print("Model score is : ", clf.score(x_test, y_test))
# print(np.mean(y_test == y_pred))
pred_probality = clf.predict_proba(x_test)
print("Predict probability : ", pred_probality)
# cross_validation score
cross_validation_score = cross_val_score(clf, x_train, y_train, cv=6)
print("Cross validation score : ", cross_validation_score)
cross_validation_predict = cross_val_predict(clf, x_train, y_train, cv=6)
print("Cross validation predict : ", cross_validation_score)
cross_val_accuracy = np.mean(cross_validation_score) * 100
print("cross validation accuracy : ", cross_val_accuracy)
# ROC
print("pred_probality : ", pred_probality, "length of prediction prob : ", len(pred_probality))
y_probs_positive = pred_probality[:, 1]
print("y_probs_positive : ", y_probs_positive)
fpr, tpr, thresholds = roc_curve(y_test, y_probs_positive)
print("fpr : ", fpr)
print("roc_auc_score : ", roc_auc_score(y_test, y_probs_positive))
plt.plot(fpr, tpr, color="orange", label="ROC")
plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--", label="Gussing")
plt.xlabel("False positive rate (fpr)")
plt.ylabel("True positive rate (tpr)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.show()
