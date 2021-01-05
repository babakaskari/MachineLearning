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
import evaluator
from sklearn.metrics import roc_curve
import seaborn as sns
from sklearn import metrics
sns.set()


dataset = prepossessed_dataset.just_labeled_splitter(0.2)

print("new dataset  :  \n ", dataset)
labeled_x_train = dataset["labeled_x_train"]
labeled_y_train = dataset["labeled_y_train"]
labeled_x_test = dataset["labeled_x_test"]
labeled_y_test = dataset["labeled_y_test"]

dataset = prepossessed_dataset.unlabeled_splitter(0.2)

unlabeled_x_train = dataset["unlabeled_x_train"]
unlabeled_x_test = dataset["unlabeled_x_test"]

clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=20, random_state=42, n_jobs=-1)
clf.fit(labeled_x_train, labeled_y_train)
plt.close("all")
evaluator.evaluate_preds(clf, labeled_x_train, labeled_y_train, labeled_x_test, labeled_y_test)

y_preds = clf.predict(unlabeled_x_test)

print("predicted labels : \n", y_preds)
