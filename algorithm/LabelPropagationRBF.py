import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
from matplotlib import dates as mpl_dates
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import OneHotEncoder
import sklearn
import seaborn as sns
pd.set_option('mode.chained_assignment', None)

df = pd.read_csv("../dataset/Acoustic Logger Data.csv")
###################################################### COLUMN SELECTION LVL & SPR
df_take_Lvl = df.loc[df["LvlSpr"] == "Lvl"]
df_take_Spr = df.loc[df["LvlSpr"] == "Spr"]
###################################################### MELT THE MENTIONED COLUMNS WITH ID & DATE
df_date_Lvl = pd.melt(df_take_Lvl, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(), var_name='Date')
df_date_Spr = pd.melt(df_take_Spr, id_vars=['LvlSpr', 'ID'], value_vars=df.loc[:0, '02-May':].columns.values.tolist(), var_name='Date')
df_merge_LvlSpr = pd.merge(df_date_Lvl, df_date_Spr, on= ['ID', 'Date'], suffixes=("_Lvl", "_Spr"))
df_drop_LvlSpr = df_merge_LvlSpr.drop(['LvlSpr_Lvl', 'LvlSpr_Spr'], axis=1).dropna()
df_drop_LvlSpr['Date'] = pd.to_datetime(df_drop_LvlSpr['Date'], format='%d-%b')
df_drop_LvlSpr['Date'] = df_drop_LvlSpr['Date'].dt.strftime('%d-%m')

df7 = pd.read_csv("../dataset/Leak Alarm Results.csv")
df7['Date Visited'] = pd.to_datetime(df7['Date Visited'], format='%d/%m/%Y')
df7['Date Visited'] = df7['Date Visited'].dt.strftime('%d-%m')
df_change_column_name = df7.rename(columns={'Date Visited': 'Date'})

df8_merge = pd.merge(df_drop_LvlSpr, df_change_column_name, on=['ID', 'Date'], how='left')
df8_sort = df8_merge.sort_values(['Leak Alarm', 'Leak Found']).reset_index(drop=True)
df8_sort["Leak Alarm"] = df8_sort["Leak Alarm"].fillna("N")

print('OUR FIRST REARRANGED DATASET IS:')
print(df8_sort)
#################################################### ONE HOT ENCODING
columns_to_OHE = df8_sort[['Date', 'Leak Alarm']]
df11_selected_columns = df8_sort[['ID', 'value_Lvl','value_Spr', 'Leak Found']]
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(columns_to_OHE)
ohe_df = pd.DataFrame(data=onehot_encoded, index=[i for i in range(onehot_encoded.shape[0])],
                      columns=['f'+str(i) for i in range(onehot_encoded.shape[1])])
df8_sort = ohe_df.join(df11_selected_columns)
df9 = df8_sort[["Leak Found"]]
df10 = df8_sort.loc[df9['Leak Found'].notnull()]
# df9["Leak Found"] = df9["Leak Found"].fillna("-1")
# df8.to_csv('OHE.csv')
df10 = df10.drop(labels=['Leak Found'], axis=1)
df12 = df8_sort.drop(labels=['Leak Found'], axis=1)

################################################### USING SAFE RBF KERNEL FUNCTION FOR GAMMA VARIATION

def rbf_kernel_safe(X, Y=None, gamma=None):
    X, Y = sklearn.metrics.pairwise.check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = sklearn.metrics.pairwise.euclidean_distances(X, Y, squared=True)
    K *= -gamma
    K -= K.max()
    np.exp(K, K)  # exponentiate K in-place
    return K

################################################### LABEL PROPAGATION WITH RBF ALGORITHMS
def label_prop():

    labels = df9.loc[df9['Leak Found'].notnull(), ['Leak Found']]
    model = LabelPropagation(kernel=rbf_kernel_safe)
    model.fit(df10, labels.values.ravel())
    pred = np.array(model.predict(df12))
    df13 = pd.DataFrame(pred, columns=['Prediction'])
    df14 = pd.concat([df12, df13], axis=1)
    print(df14[['ID', 'Prediction']])
    # print(df14.loc[df14['Prediction'] == 'Y'])
    plt.style.use ( 'seaborn' )
    df14['Prediction'].value_counts().plot(kind='bar')
    plt.xticks ( [ 0 , 1 , 2 ] , [ 'NO' , 'YES' , 'N-PRV' ] )
    plt.ylabel('Number of occurrences after prediction by RBF algorithm');
    plt.show()



if __name__ == '__main__':
    label_prop()

################################################## PLOTTING THE REQUIRED COLUMNS

# My_dataset = pd.read_csv('beforeOHE.csv')
# My_dataset.value_Lvl
# print(My_dataset.value_Lvl)

# sns.set(font_scale=1.4)
# My_dataset['value_Lvl'].plot(kind='hist');
# plt.xlabel("Date", labelpad=2)
# plt.ylabel("Average Level (value_Lvl)", labelpad=2)
# plt.title("Distribution of value_Lvl", y=1.015, fontsize=22);
# plt.show()



######################################################## plot
# plt.style.use('seaborn')
#
# data = pd.read_csv('beforeOHE.csv')
# data['Date'] = pd.to_datetime(data['Date'], format='%d-%m')
#
# data.sort_values('Date', inplace=True)
# price_date = data['Date']
# price_close = data['value_Lvl']
# plt.plot_date(price_date, price_close, linestyle='solid')
# plt.gcf().autofmt_xdate()
# date_format = mpl_dates.DateFormatter('%d-%m-%Y')
# plt.gca().xaxis.set_major_formatter(date_format)
# plt.tight_layout()
# plt.title('captured average level (Lvl) histogram on available dates')
# plt.xlabel('Date')
# plt.ylabel('Average level (Lvl)')
# plt.show()

################################################## Histogram for Lvl & Sor
# _ = plt.hist(data['value_Spr'].ravel(), bins='auto')  # arguments are passed to np.histogram
# plt.title("Spread of noise (X-axis) Histogram based on number of occurrences in Y axis ")
#
# plt.show()

############################################ plot after prediction
# data['Leak Found'].value_counts().plot(kind='bar')
# plt.xticks([0,1,2], ['NO', 'YES', 'N-PRV'])
# plt.ylabel('Count');
# plt.show()