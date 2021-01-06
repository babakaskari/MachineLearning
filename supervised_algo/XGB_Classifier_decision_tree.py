import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
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
from sklearn.neighbors import KNeighborsClassifier
# import os
# os.environ["PATH"] += os.pathsep + 'C:\Program Files\Graphviz\bin'
import sys
import prepossessed_dataset
import evaluator
from sklearn.metrics import roc_curve
import seaborn as sns
import xgboost as xgb
from sklearn import metrics
from xgboost import XGBClassifier
sys.path.insert(0, "D:\\Graphviz\\bin")
sys.path.insert(0, "D:\\Graphviz")
sns.set()

dataset = prepossessed_dataset.labeled()
x_train = dataset["x_train"]
y_train = dataset["y_train"]
x_test = dataset["x_test"]
y_test = dataset["y_test"]
x_cv = dataset["x_cv"]
y_cv = dataset["y_cv"]

# clf = XGBClassifier()
clf = xgb.sklearn.XGBClassifier(nthread=-1, n_estimators=50, seed=42)
clf.fit(x_train, y_train)
xgb_pred = clf.predict(x_train)
evaluator.evaluate_preds(clf, x_train, y_train, x_test, y_test, x_cv, y_cv)
plt.figure(figsize=(20, 15))
xgb.plot_importance(clf, ax=plt.gca())
plt.show()
plt.figure(figsize=(20, 15))
xgb.plot_tree(clf, ax=plt.gca())
plt.show()
print("Number of boosting trees: {}".format(clf.n_estimators))
print("Max depth of trees: {}".format(clf.max_depth))
print("Objective function: {}".format(clf.objective))
