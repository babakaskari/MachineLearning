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

# print('Number of data points in train data:', x_train.shape[0])
# print('Number of data points in test data:', x_test.shape[0])
# print('Number of data points in test data:', x_cv.shape[0])
clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=20, random_state=42, n_jobs=-1)
clf.fit(x_train, y_train)
prepossessed_dataset.evaluate_preds(clf, x_train, y_train, x_test, y_test, x_cv, y_cv)
