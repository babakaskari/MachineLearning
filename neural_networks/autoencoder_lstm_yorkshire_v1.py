from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
tf.enable_eager_execution()
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import auc, roc_curve
import pandas as pd
import yaml
from keras import layers
import autoencoder_model
import neural_network_evaluator
import visualiser
# import prepossessed_dataset
from prepossessed_dataset import labeled
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import pickle
sns.set()
# import neural_network_evaluator
# import visualiser

dataset = labeled()
# print("dataset: ", dataset["dataset"])
df = dataset["dataset"].drop(['ID', 'Date'], axis=1)
dataset = df.reset_index(drop=True)
dataset.rename({'Leak Found': 'label'}, axis=1, inplace=True)
df_label = dataset.drop(['value_Lvl', 'value_Spr'], axis=1)
df_data = dataset.drop(['label'], axis=1)
min_max_scaler = MinMaxScaler()
df_data = min_max_scaler.fit_transform(df_data)
df_dataset = pd.DataFrame(df_data, columns=['value_Lvl', 'value_Spr'])
print("df_dataset : ", df_dataset)
dataset = pd.concat([df_dataset, df_label], axis=1)
print("dataset : ", dataset)
with open("initializer.yaml") as stream:
    param = yaml.safe_load(stream)

threshold_lstm = param["threshold_lstm"]
LABELS = param["LABEL"]

x_normal = dataset[dataset["label"] == 0]
x_abnormal = dataset[dataset["label"] == 1].reset_index(drop=True)
# print("x_normal : ", x_normal)
# print("x_abnormal : ", x_abnormal)
x_test_normal = x_normal.iloc[0:x_abnormal.shape[0]]
train = x_normal.iloc[x_abnormal.shape[0]:].reset_index(drop=True)
# print("x_test_normal : ", x_test_normal)
test = pd.concat([x_test_normal, x_abnormal], axis=0, ignore_index=True)

# print("train : ", train)
# print("test : ", test)
# print(train.shape, test.shape)
test, valid = train_test_split(test,
                               test_size=param["lstm_data_split"]["test_size"],
                               shuffle=param["lstm_data_split"]["shuffle"],
                               random_state=param["lstm_data_split"]["random_state"])
y_train = train["label"]
x_train = train.drop(['label'], axis=1)
y_valid = valid["label"]
x_valid = valid.drop(['label'], axis=1)
y_test = test["label"]
x_test = test.drop(['label'], axis=1)

# print(x_train.shape, x_test.shape)
x_train = x_train.to_numpy()
x_train_scaled = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_valid = x_valid.to_numpy()
x_valid_scaled = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 1))
x_test = x_test.to_numpy()
x_test_scaled = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# print("x_train shape :   \n", x_train.shape)
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# print(y_test)
input_dim1 = x_train_scaled.shape[1]
input_dim2 = x_train_scaled.shape[2]
print(input_dim1)
print(input_dim2)

# ######################################

autoencoder = autoencoder_model.autoencoder_model(input_dim1, input_dim2)

print("output_shape  :   ", autoencoder.output_shape)

# Model summary
autoencoder.summary()
# ######################################
autoencoder = autoencoder_model.autoencoder_lstm(input_dim1, input_dim2)
autoencoder.compile(**param["fit_lstm"]["compile"])
history = autoencoder.fit(
            x_train_scaled,
            x_train_scaled,
            epochs=param["fit_lstm"]["epochs"],
            batch_size=param["fit_lstm"]["batch_size"],
            verbose=param["fit_lstm"]["verbose"],
            validation_split=param["fit_lstm"]["validation_split"],
            # validation_data=(x_test, x_test),
            shuffle=param["fit_lstm"]["shuffle"])


neural_network_evaluator.evaluate_ann(history)
visualiser.lstm_train_val_loss_plotter(history, param["fit_lstm"]["epochs"])
# ###########################


def flatten(x):
    flattened_x = np.empty((x.shape[0], x.shape[2]))
    for i in range(x.shape[0]):
        flattened_x[i] = x[i, (x.shape[1] - 1), :]
    return flattened_x


# #########################

x_valid_predictions = autoencoder.predict(x_valid_scaled)
# mse = np.mean(np.power(flatten(x_valid) - flatten(x_valid_predictions), 2), axis=1)
mse = np.mean(np.power(flatten(x_valid_scaled) - flatten(x_valid_predictions), 2), axis=1)

# ###################
error_df = pd.DataFrame({'Reconstruction_error': mse,
                         'True_class': valid['label']})
# ###############################
visualiser.precision_recall_plotter(error_df)
# ##############################
x_test_predictions = autoencoder.predict(x_test_scaled)
visualiser.test_plotter(threshold_lstm, flatten(x_test_scaled), flatten(x_test_predictions), test['label'])
# ###################################################
visualiser.confusion_matrix_plotter(threshold_lstm, LABELS, error_df)
# ###################################
visualiser.auc_roc_curved(error_df)
# ##################################################



