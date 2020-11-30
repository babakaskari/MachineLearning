import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import OneHotEncoder
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
sns.set()


def main():

    pd.set_option('mode.chained_assignment', None)

    df = pd.read_csv("./dataset/Acoustic Logger Data.csv")
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

    df7 = pd.read_csv("./dataset/Leak Alarm Results.csv")
    df7['Date Visited'] = pd.to_datetime(df7['Date Visited'], format='%d/%m/%Y')
    df7['Date Visited'] = df7['Date Visited'].dt.strftime('%d-%m')
    df7 = df7.rename(columns={'Date Visited': 'Date'})

    df8 = pd.merge(df6, df7, on=['ID', 'Date'], how='left')
    df8 = df8.sort_values(['Leak Alarm', 'Leak Found']).reset_index(drop=True)
    # df8["Leak Alarm"] = df8["Leak Alarm"].fillna(-1)
    df8["Leak Found"] = df8["Leak Found"].fillna(-1)
    dataset = df8
    dataset["Leak Found"].replace(["Y", "N", "N-PRV"], [1, 0, -2], inplace=True)
    # dataset["Leak Alarm"].replace(["Y", "N"], [1, 0], inplace=True)
    dataset1 = dataset
    dataset = dataset1.drop(['Leak Alarm'], axis=1)

    dataset['Date'] = dataset['Date'].str.replace('\D', '').astype(int)
    print(dataset)
    print(dataset.isna().sum())
    # corrolation matrix
    print("Features : ")
    print(dataset.columns.values)

    df = pd.DataFrame(dataset, columns=['Date', 'ID', 'value_Lvl', 'value_Spr', 'Leak Found'])
    corrMatrix = df.corr()
    sns.heatmap(corrMatrix, annot=True, cmap="YlGnBu")
    plt.show()
    leak_found = dataset["Leak Found"]
    dataset3 = dataset.drop(['Leak Found'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(dataset3, leak_found, stratify=leak_found, test_size=0.2, random_state=42)
    x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, stratify=y_train, test_size=0.2, random_state=42)
    print('Number of data points in train data:', x_train.shape[0])
    print('Number of data points in test data:', x_test.shape[0])
    print('Number of data points in test data:', x_cv.shape[0])

    data_dict = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "x_cv": x_cv,
        "y_cv": y_cv

    }

    return data_dict


if __name__ == "__main__":
    main()




