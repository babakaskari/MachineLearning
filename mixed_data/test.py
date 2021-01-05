import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import OneHotEncoder
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import preprocessing, metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
from gaussrank import *
import warnings
import prepossessed_dataset
import seaborn as sns
sns.set()

plt.close("all")
dataset = prepossessed_dataset.unlabeled_splitter(0.2)
print("new dataset  :  \n ", dataset["unlabeled_x_train"])
print("new dataset shape :  \n ", dataset["unlabeled_x_train"].shape)
print("new dataset  :  \n ", dataset["unlabeled_x_test"])
print("new dataset shape :  \n ", dataset["unlabeled_x_test"].shape)

