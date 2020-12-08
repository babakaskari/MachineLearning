
# importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.linear_model import BayesianRidge, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import warnings

warnings.filterwarnings('ignore')

# preprocessing

# ## ###########################
df = pd.read_csv("../dataset/Acoustic Logger Data.csv")
df1 = df.loc[df["LvlSpr"] == "Lvl"]
df3 = df.loc[df["LvlSpr"] == "Spr"]
df2 = pd.melt(df1, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(), var_name='Date')
df4 = pd.melt(df3, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(), var_name='Date')
df5 = pd.merge(df2, df4, on= ['ID', 'Date'], suffixes=("_Lvl", "_Spr"))
df6 = df5.drop(['LvlSpr_Lvl', 'LvlSpr_Spr'], axis=1).dropna()
df6['Date'] = pd.to_datetime(df6['Date'], format='%d-%b')
df6['Date'] = df6['Date'].dt.strftime('%d-%m')

df7 = pd.read_csv("../dataset/Leak Alarm Results.csv")
df7['Date Visited'] = pd.to_datetime(df7['Date Visited'], format='%d/%m/%Y')
df7['Date Visited'] = df7['Date Visited'].dt.strftime('%d-%m')
df7 = df7.rename(columns={'Date Visited': 'Date'})

df8 = pd.merge(df6, df7, on=['ID', 'Date'], how='left')
df8 = df8.sort_values(['Leak Alarm', 'Leak Found']).reset_index(drop=True)
# df8["Leak Alarm"] = df8["Leak Alarm"].fillna(-1)
#df8["Leak Found"] = df8["Leak Found"].fillna(-1)
dataset = df8
###################################################### Delete these row indexes from dataFrame
indexNames = dataset[dataset['Leak Found'] == 'N-PRV'].index
dataset.drop(indexNames, index=None, inplace=True)
dataset.reset_index(drop=True, inplace=True)
###################################################### DROPPING LEAK ALARM & LEAK FOUND
dataset["Leak Found"].replace(["Y", "N"], [1, 0], inplace=True)
# dataset["Leak Alarm"].replace(["Y", "N"], [1, 0], inplace=True)
dataset = dataset.drop(['Leak Alarm'], axis=1)

# ############################################################ Convert Date categorical to numerical
# dataset['Date'] = dataset['Date'].str.replace('\D', '').astype(int)
date_encoder = preprocessing.LabelEncoder()
date_encoder.fit(dataset['Date'])
# print(list(date_encoder.classes_))
dataset['Date'] = date_encoder.transform(dataset['Date'])
# print(dataset.to_string(max_rows=200))
print("Number of null values in dataset :\n", dataset.isna().sum())
# ##################################################### CORRELATION MATRIX
# print(dataset.columns.values)
# dataset2 = dataset.drop(["Leak Found"], axis=1)
# df = pd.DataFrame(dataset2, columns=['Date', 'ID', 'value_Lvl', 'value_Spr'])
# corrMatrix = df.corr()
# sns.heatmap(corrMatrix, annot=True, cmap="YlGnBu")
# plt.show()
# ##################################################### SPLIT THE DATASET

x_train = dataset.loc[dataset['Leak Found'].notna()]
y_train = x_train.drop(["Leak Found"], axis=1)
# x_train = x_train.sample(frac=1)
x_test = dataset.loc[dataset['Leak Found'].notna()]
y_test = x_test.drop(["Leak Found"], axis=1)
print()
feature = x_train.columns
targets = 'Leak Alarm'
########################

model_factory = [
    LogisticRegression(),
    KNeighborsClassifier(),
    BaggingClassifier(n_estimators=100),
    XGBRegressor(nthread=1),
    GradientBoostingRegressor(),
    RandomForestClassifier(),
    BayesianRidge(),
]


"""
model_factory = [
    RandomForestRegressor(),
    XGBRegressor(nthread=1),
    Ridge(),
    BayesianRidge(),
    ExtraTreesRegressor(),
    ElasticNet(),
    KNeighborsRegressor(),
    GradientBoostingRegressor()
]

"""

# #######################################


for model in model_factory:
    model.seed = 42
    num_folds = 3

    scores = cross_val_score(model, x_train, y_train, cv=num_folds, scoring='neg_mean_squared_error')
    # print("Score is : ", scores)

    score_description = " %0.2f (+/- %0.2f)" % (np.sqrt(scores.mean() * -1), scores.std() * 2)

    print('{model:25} CV-5 RMSE: {score}'.format(
        model=model.__class__.__name__,
        score=score_description
    ))

# ////////////////////////////////


class PseudoLabeler(BaseEstimator, RegressorMixin):
    """

    Sci-kit learn wrapper for creating pseudo-lebeled estimators.

    """

    def __init__(self, models, unlabled_data, features, target, sample_rate=0.2, seed=42):

        """
        @sample_rate - percent of samples used as pseudo-labelled data
         from the unlabelled dataset
        """
        assert sample_rate <= 1.0, 'Sample_rate should be between 0.0 and 1.0.'

        self.sample_rate = sample_rate
        self.seed = seed
        self.model = models
        self.model.seed = seed

        self.unlabled_data = unlabled_data
        self.features = features
        self.target = target

    def get_params(self, deep=True):

        return {
            "sample_rate": self.sample_rate,
            "seed": self.seed,
            "model": self.model,
            "unlabled_data": self.unlabled_data,
            "features": self.features,
            "target": self.target
             }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            return self

    def fit(self, X, y):
        """
        Fit the data using pseudo labeling.
        """
        augemented_train = self.__create_augmented_train(X, y)
        self.model.fit(
                        augemented_train[self.features],
                        augemented_train[self.target],
                        )
        return self

    def __create_augmented_train(self, X, y):

        """
         Create and return the augmented_train set that consists
         of pseudo-labeled and labeled data.
        """
        num_of_samples = int(len(self.unlabled_data) * self.sample_rate)

        # Train the model and creat the pseudo-labels
        self.model.fit(X, y)
        pseudo_labels = self.model.predict(self.unlabled_data[self.features])

        # Add the pseudo-labels to the test set
        pseudo_data = self.unlabled_data.copy(deep=True)
        pseudo_data[self.target] = pseudo_labels

        # Take a subset of the test set with pseudo-labels and append in onto
        # the training set
        sampled_pseudo_data = pseudo_data.sample(n=num_of_samples)
        temp_train = pd.concat([X, y], axis=1)
        augemented_train = pd.concat([sampled_pseudo_data, temp_train])

        return shuffle(augemented_train)

    def predict(self, X):
        """
        Returns the predicted values.
        """
        return self.model.predict(X)

    def get_model_name(self):
        return self.model.__class__.__name__


model_factory = [
    # XGBRegressor(nthread=1),
    KNeighborsClassifier(),
    PseudoLabeler(
        # XGBRegressor(nthread=1),
        KNeighborsClassifier(),
        x_test,
        feature,
        targets,
        sample_rate=0.3
    ),
]

# /////////////////////////////////////////////////

for model in model_factory:
    model.seed = 42
    num_folds = 8

    scores = cross_val_score(model, x_train, y_train, cv=num_folds, scoring='neg_mean_squared_error', n_jobs=8)
    score_description = "MSE: %0.4f (+/- %0.4f)" % (np.sqrt(scores.mean() * -1), scores.std() * 2)

print('{model:25} CV-{num_folds} {score_cv}'.format(
    model=model.__class__.__name__,
    num_folds=num_folds,
    score_cv=score_description
))

# ////////////////////////

