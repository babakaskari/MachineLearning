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


def just_labeled():
    df = pd.read_csv("../dataset/Acoustic Logger Data.csv")
    df1 = df.loc[df["LvlSpr"] == "Lvl"]
    df3 = df.loc[df["LvlSpr"] == "Spr"]
    df2 = pd.melt(df1, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(),
                  var_name='Date')
    df4 = pd.melt(df3, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(),
                  var_name='Date')
    df5 = pd.merge(df2, df4, on=['ID', 'Date'], suffixes=("_Lvl", "_Spr"))
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
    # df8["Leak Found"] = df8["Leak Found"].fillna(-1)
    dataset = df8

    # ##################################################### Delete these row indexes from dataFrame
    indexNames = dataset[dataset['Leak Found'] == 'N-PRV'].index
    dataset.drop(indexNames, index=None, inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    # ##################################################### DROPPING LEAK ALARM & LEAK FOUND
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
    # ##################################################### SPLIT THE DATASET
    x_labeled_data = dataset.loc[dataset['Leak Found'].notna()]

    y_labeled_data = x_labeled_data["Leak Found"]
    x_labeled_data = x_labeled_data.drop(["Leak Found"], axis=1)

    # ############################################## standard scaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(x_labeled_data)
    x_train = pd.DataFrame(data_scaled)
    # print("x_train after normalization : ", x_train.head())
    # print("x_train description after normalization: ", x_labeled_data.describe())

    # ############################################################
    # x_train = x_train.sample(frac=1)
    x_unlabeled_data = dataset.loc[dataset['Leak Found'].isna()]
    y_unlabeled_data = x_unlabeled_data.drop(["Leak Found"], axis=1)
    print("ID in just_labeld dtaaset : ", x_labeled_data["ID"].nunique())
    print("ID in just_labeld dtaaset : ", x_labeled_data["ID"].unique())
    print("number of occurence : ", x_labeled_data['ID'].value_counts())

    x_train, x_test, y_train, y_test = train_test_split(x_labeled_data,
                                                        y_labeled_data,
                                                        test_size=0.2,
                                                        random_state=42)
    x_train, x_cv, y_train, y_cv = train_test_split(x_train,
                                                    y_train,
                                                    stratify=y_train,
                                                    test_size=0.2)

    data_dict = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "x_cv": x_cv,
        "y_cv": y_cv

    }

    return data_dict


def unlabeled():

    df = pd.read_csv("../dataset/Acoustic Logger Data.csv")
    df1 = df.loc[df["LvlSpr"] == "Lvl"]
    df3 = df.loc[df["LvlSpr"] == "Spr"]
    df2 = pd.melt(df1, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(),
                  var_name='Date')
    df4 = pd.melt(df3, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(),
                  var_name='Date')
    df5 = pd.merge(df2, df4, on=['ID', 'Date'], suffixes=("_Lvl", "_Spr"))
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
    # df8["Leak Found"] = df8["Leak Found"].fillna(-1)
    dataset = df8

    # ##################################################### Delete these row indexes from dataFrame
    indexNames = dataset[dataset['Leak Found'] == 'N-PRV'].index
    dataset.drop(indexNames, index=None, inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    # ##################################################### DROPPING LEAK ALARM & LEAK FOUND
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
    x_train = dataset.loc[dataset['Leak Found'].isna()]
    x_train = x_train.drop(["Leak Found"], axis=1)
    # x_train = x_train.sample(frac=1)
    x_test = dataset.loc[dataset['Leak Found'].notna()]
    y_test = x_test.loc[dataset['Leak Found'].notna(), ['Leak Found']]
    # ##################################################### CORRELATION OF KNOWN LABELLED DATA
    df = pd.DataFrame(x_test, columns=['Date', 'ID', 'value_Lvl', 'value_Spr', 'Leak Found'])
    corrMatrix = df.corr()
    sns.heatmap(corrMatrix, annot=True, cmap="YlGnBu")
    plt.show()
    # #####################################################
    x_test = x_test.drop(["Leak Found"], axis=1)
    # print("x_test shape is equal to :  ", x_test.shape)
    # print("dataset features :  ", dataset.columns)
    # ############################################# CREATING DUMMY_DATA
    # x_centroid = np.array(x_test.iloc[[16, 17], ])
    dummy_data = dataset.drop(['Leak Found'], axis=1)
    print("Description  : \n ", dummy_data.describe())
    # ############################################# TO TAKE THE SELECTED SAMPLE FOR OUR XTRAIN
    # dummy_data = dummy_data.sample(frac=1)
    # x_dummy = dummy_data[:54]
    # x_train = x_dummy
    # ############################################ SCALER NORMALIZATION   " TO BE MODIFIED LATER"

    # scaler = MinMaxScaler()
    # # fit using the train set
    # scaler.fit(x_train)
    # # transform the test test
    # x_train = scaler.transform(x_train)
    # # build the scaler model
    # scaler = Normalizer()
    #
    # # fit using the train set
    # scaler.fit(x_train)
    # # transform the test test
    # x_train = scaler.transform(x_train)
    # plt.show()
    ########################################### APPLYING GUASSRANK NORMALIZATION
    """
    x_cols = x_train.columns[:]
    x = x_train[x_cols]

    s = GaussRankScaler()
    x_ = s.fit_transform( x )
    assert x_.shape == x.shape
    x_train[x_cols] = x_
    """
    ############################################### standard scaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(data_scaled)
    print("x_train description : \n", x_train.describe())
    ########################################## TO REPRESENT OUR DATASET, ALL COLUMNS IN MATRIX FORM
    x_train = pd.DataFrame(x_train)

    data_dict = {
        "x_train": x_train,
        "x_test": x_test,
    }

    return data_dict


def labeled():
    df = pd.read_csv("../dataset/Acoustic Logger Data.csv")
    df1 = df.loc[df["LvlSpr"] == "Lvl"]
    df3 = df.loc[df["LvlSpr"] == "Spr"]
    df2 = pd.melt(df1, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(),
                  var_name='Date')
    df4 = pd.melt(df3, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(),
                  var_name='Date')
    df5 = pd.merge(df2, df4, on=['ID', 'Date'], suffixes=("_Lvl", "_Spr"))
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
    df8["Leak Found"] = df8["Leak Found"].fillna(0)

    dataset = df8

    indexNames = dataset[dataset['Leak Found'] == 'N-PRV'].index
    # Delete these row indexes from dataFrame
    dataset.drop(indexNames, index=None, inplace=True)
    dataset.reset_index(inplace=True)
    dataset["Leak Found"].replace(["Y", "N"], [1, 0], inplace=True)
    # dataset["Leak Alarm"].replace(["Y", "N"], [1, 0], inplace=True)
    dataset1 = dataset
    dataset = dataset1.drop(['Leak Alarm'], axis=1)

    # ############################################################ Convert Date categorical to numerical
    # dataset['Date'] = dataset['Date'].str.replace('\D', '').astype(int)
    date_encoder = preprocessing.LabelEncoder()
    date_encoder.fit(dataset['Date'])
    # print(list(date_encoder.classes_))
    dataset['Date'] = date_encoder.transform(dataset['Date'])
    # print(dataset.to_string(max_rows=200))
    dataset = dataset.drop_duplicates()
    print(" dataset description : \n", dataset.describe())
    # ##############################################
    dataset = dataset.drop(['index'], axis=1)
    # print(("df8 shape : ", df8.shape))

    # corrolation matrix
    # print(dataset.columns.values)
    df = pd.DataFrame(dataset, columns=['Date', 'ID', 'value_Lvl', 'value_Spr'])
    corrMatrix = df.corr()
    sns.heatmap(corrMatrix, annot=True, cmap="YlGnBu")
    plt.show()

    # dataset = dataset.loc[:80]
    dataset = dataset.sample(frac=1)

    print("Number of null values in dataset : \n", dataset.isna().sum())
    print("datase shape before duplicate : ", dataset.shape)

    leak_found = dataset.drop(['ID', 'Date', 'value_Lvl', 'value_Spr'], axis=1)
    dataset2 = dataset.drop(['Leak Found'], axis=1)
    dataset5 = dataset2.drop_duplicates()

    print("datase shape after duplicate : ", dataset5.shape)
    print("ID in whole dtaaset : ", dataset2["ID"].nunique())
    print("ID in just_labeld dtaaset : ", dataset2["ID"].unique())
    print("number of occurence : ", dataset2['ID'].value_counts())

    # print("leak found shape : ", leak_found.shape)
    # print("dataset2 shape : ", dataset2.shape)
    # ########################################## APPLYING GUASSRANK NORMALIZATION

    x_cols = dataset2.columns[:]
    x = dataset2[x_cols]

    s = GaussRankScaler()
    x_ = s.fit_transform(x)
    assert x_.shape == x.shape
    dataset2[x_cols] = x_
    print("GaussRankScaler dataset description :\n ", dataset2.describe())

    # ############################################## standard scaler
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(data_scaled)
    print("x_train description : ", x_train.describe())
    """
    # ##############################################

    x_train, x_test, y_train, y_test = train_test_split(dataset2,
                                                        leak_found,
                                                        test_size=0.2,
                                                        random_state=42)

    x_train, x_cv, y_train, y_cv = train_test_split(x_train,
                                                    y_train,
                                                    stratify=y_train,
                                                    test_size=0.2)

    data_dict = {

        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "x_cv": x_cv,
        "y_cv": y_cv,


    }

    return data_dict


def propagation():
    df = pd.read_csv("../dataset/Acoustic Logger Data.csv")
    df1 = df.loc[df["LvlSpr"] == "Lvl"]
    df3 = df.loc[df["LvlSpr"] == "Spr"]
    df2 = pd.melt(df1, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(),
                  var_name='Date')
    df4 = pd.melt(df3, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(),
                  var_name='Date')
    df5 = pd.merge(df2, df4, on=['ID', 'Date'], suffixes=("_Lvl", "_Spr"))
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
    df8["Leak Found"] = df8["Leak Found"].fillna(-1)

    dataset = df8
    indexNames = dataset[dataset['Leak Found'] == 'N-PRV'].index
    # Delete these row indexes from dataFrame
    dataset.drop(indexNames, index=None, inplace=True)
    dataset.reset_index(inplace=True)
    dataset["Leak Found"].replace(["Y", "N"], [1, 0], inplace=True)
    # dataset["Leak Alarm"].replace(["Y", "N"], [1, 0], inplace=True)
    dataset1 = dataset
    dataset = dataset1.drop(['Leak Alarm'], axis=1)

    # ############################################################ Convert Date categorical to numerical
    # dataset['Date'] = dataset['Date'].str.replace('\D', '').astype(int)
    date_encoder = preprocessing.LabelEncoder()
    date_encoder.fit(dataset['Date'])
    # print(list(date_encoder.classes_))
    dataset['Date'] = date_encoder.transform(dataset['Date'])
    # print(dataset.to_string(max_rows=200))
    dataset = dataset.drop_duplicates()
    print(" dataset description : \n", dataset.describe())
    # ##############################################

    # corrolation matrix
    # print(dataset.columns.values)
    df = pd.DataFrame(dataset, columns=['Date', 'ID', 'value_Lvl', 'value_Spr'])
    corrMatrix = df.corr()
    sns.heatmap(corrMatrix, annot=True, cmap="YlGnBu")
    plt.show()
    tempdata = dataset
    dataset = dataset.loc[:80]
    dataset = dataset.sample(frac=1)
    # print("dataset shape: ", dataset.shape)
    print("Number of null values in dataset : \n", dataset.isna().sum())
    # print("dataset : ", dataset.shape[0])
    # dataset2 = dataset.drop(["Leak Found"], axis=1)
    dataset2 = dataset
    print("dataset features : ", dataset.columns)
    leak_found = dataset2["Leak Found"]
    dataset2 = dataset.drop(['Leak Found'], axis=1)
    # ########################################## APPLYING GUASSRANK NORMALIZATION

    x_cols = dataset2.columns[:]
    x = dataset2[x_cols]

    s = GaussRankScaler()
    x_ = s.fit_transform(x)
    assert x_.shape == x.shape
    dataset2[x_cols] = x_
    print("GaussRankScaler dataset description :\n ", dataset2.describe())

    # ############################################## standard scaler
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(data_scaled)
    print("x_train description : ", x_train.describe())
    """
    # ##############################################
    print("dataset2 features : ", dataset2.columns)

    x_train, x_test, y_train, y_test = train_test_split(dataset2,
                                                        leak_found,
                                                        test_size=0.2,
                                                        random_state=42)

    data_dict = {

        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,

    }

    return data_dict


def semi_super():
    df = pd.read_csv("../dataset/Acoustic Logger Data.csv")
    df1 = df.loc[df["LvlSpr"] == "Lvl"]
    df3 = df.loc[df["LvlSpr"] == "Spr"]
    df2 = pd.melt(df1, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(),
                  var_name='Date')
    df4 = pd.melt(df3, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(),
                  var_name='Date')
    df5 = pd.merge(df2, df4, on=['ID', 'Date'], suffixes=("_Lvl", "_Spr"))
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
    # df8["Leak Found"] = df8["Leak Found"].fillna(-1)

    dataset = df8
    indexNames = dataset[dataset['Leak Found'] == 'N-PRV'].index
    # Delete these row indexes from dataFrame
    dataset.drop(indexNames, index=None, inplace=True)
    dataset.reset_index(inplace=True)
    dataset["Leak Found"].replace(["Y", "N"], [1, 0], inplace=True)
    # dataset["Leak Alarm"].replace(["Y", "N"], [1, 0], inplace=True)
    dataset1 = dataset
    dataset = dataset1.drop(['Leak Alarm'], axis=1)

    # ############################################################ Convert Date categorical to numerical
    # dataset['Date'] = dataset['Date'].str.replace('\D', '').astype(int)
    date_encoder = preprocessing.LabelEncoder()
    date_encoder.fit(dataset['Date'])
    # print(list(date_encoder.classes_))
    dataset['Date'] = date_encoder.transform(dataset['Date'])
    # print(dataset.to_string(max_rows=200))
    dataset = dataset.drop_duplicates()
    print(" dataset description : \n", dataset.describe())
    # ##############################################

    # corrolation matrix
    # print(dataset.columns.values)
    df = pd.DataFrame(dataset, columns=['Date', 'ID', 'value_Lvl', 'value_Spr'])
    corrMatrix = df.corr()
    sns.heatmap(corrMatrix, annot=True, cmap="YlGnBu")
    # plt.show()

    dataset = dataset.sample(frac=1)
    # print("dataset shape: ", dataset.shape)
    print("Number of null values in dataset : \n", dataset.isna().sum())
    # print("dataset : ", dataset.shape[0])
    # dataset2 = dataset.drop(["Leak Found"], axis=1)
    dataset2 = dataset
    # print("dataset features : ", dataset.columns)
    # leak_found = dataset2["Leak Found"]
    # dataset2 = dataset.drop(['Leak Found'], axis=1)

    # ########################################## APPLYING GUASSRANK NORMALIZATION
    """
    x_cols = dataset2.columns[:]
    x = dataset2[x_cols]

    s = GaussRankScaler()
    x_ = s.fit_transform(x)
    assert x_.shape == x.shape
    dataset2[x_cols] = x_
    print("GaussRankScaler dataset description :\n ", dataset2.describe())
    """
    # ############################################## standard scaler
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(data_scaled)
    print("x_train description : ", x_train.describe())
    """
    # ##############################################
    # print("dataset2 features : ", dataset2.columns)

    # print(dataset2)
    labeled_df = dataset2.loc[dataset2['Leak Found'].notnull()]
    unlabeled_df = dataset2.loc[dataset2['Leak Found'].isnull()]
    shuffled_labeled_df = labeled_df.sample(frac=1).reset_index(drop=True)
    labels = shuffled_labeled_df[["Leak Found"]]
    # df8.to_csv('OHE.csv')
    x_labeled = shuffled_labeled_df.drop(labels=['Leak Found'], axis=1)  # Labled train
    x_labeled = x_labeled.drop(labels=['index'], axis=1)
    x_labeled.reset_index(drop=True, inplace=True)
    x_unlabeled = unlabeled_df.drop(labels=['Leak Found'], axis=1)  # Unlabled
    x_unlabeled = x_unlabeled.drop(labels=['index'], axis=1)
    x_unlabeled.reset_index(drop=True, inplace=True)
    # print("X_unlabeld \n", X_unlabeled)
    # =======================labeled data
    x_cols = x_labeled.columns[:]
    x = x_labeled[x_cols]

    s = GaussRankScaler()
    x_ = s.fit_transform(x)
    assert x_.shape == x.shape
    x_labeled[x_cols] = x_

    # ===================== unlabeled data

    x_cols = x_unlabeled.columns[:]
    x = x_unlabeled[x_cols]

    s = GaussRankScaler()
    x_ = s.fit_transform(x)
    assert x_.shape == x.shape
    x_unlabeled[x_cols] = x_

    # #################################################
    test_ind = round(len(x_labeled) * 0.70)
    train_ind = test_ind + round(len(x_labeled) * 0.30)
    x_test = x_labeled.iloc[:test_ind]
    x_train = x_labeled.iloc[test_ind:train_ind]
    y_test = labels.iloc[:test_ind]
    y_train = labels.iloc[test_ind:train_ind]
    # print("X_unlabeld features  ", x_unlabeled.columns)
    # print("X_unlabeld \n", X_unlabeled)
    data_dict = {
                "x_unlabeled": x_unlabeled,
                "x_train": x_train,
                "y_train": y_train,
                "x_test": x_test,
                "y_test": y_test,

    }

    return data_dict


