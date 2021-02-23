from tensorflow.keras.layers import Input, Dense
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras import layers


def anomaly_detector(input):
    inputlayer = Input(shape=(input,))
    model_layer = Dense(8, activation="relu")(inputlayer)
    model_layer = Dense(6, activation="relu")(model_layer)
    model_layer = Dense(6, activation="relu")(model_layer)
    model_layer = Dense(4, activation="relu")(model_layer)
    model_layer = Dense(2, activation="relu")(model_layer)
    model_layer = Dense(1, activation="relu")(model_layer)
    model_layer = Dense(2, activation="relu")(model_layer)
    model_layer = Dense(4, activation="relu")(model_layer)
    model_layer = Dense(6, activation="relu")(model_layer)
    model_layer = Dense(6, activation="relu")(model_layer)
    model_layer = Dense(8, activation="relu")(model_layer)
    model_layer = Dense(input, activation="linear")(model_layer)
    # print("model_layer : ", model_layer)
    return Model(inputs=inputlayer, outputs=model_layer)


def encoder_layer(input):
    inputlayer = Input(shape=(input,))
    model_layer = Dense(64, activation="relu")(inputlayer)
    model_layer = Dense(64, activation="relu")(model_layer)
    model_layer = Dense(8, activation="relu")(model_layer)
    return Model(inputs=inputlayer, outputs=model_layer)


def decoder_layer(input):
    inputlayer = Input(shape=(input,))
    model_layer = Dense(64, activation="relu")(inputlayer)
    model_layer = Dense(64, activation="relu")(model_layer)
    model_layer = Dense(8, activation="sigmoid")(model_layer)
    print("model_layer : ", model_layer)
    return Model(inputs=inputlayer, outputs=model_layer)


class AnomalyDetector(Model):
    def __init__(self, input_dim):
        super(AnomalyDetector, self).__init__()
        self.encoder = encoder_layer(input_dim)
        self.decoder = decoder_layer(input_dim)

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AnomalyDetectorAutoencoder(Model):
    def __init__(self, input_dim):
        super(AnomalyDetectorAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
          layers.Dense(8, activation="relu", input_shape=(input_dim,)),
          layers.Dense(6, activation="relu"),
          layers.Dense(6, activation="relu"),
          layers.Dense(4, activation="relu"),
          layers.Dense(2, activation="relu")])
        self.decoder = tf.keras.Sequential([
          layers.Dense(4, activation="relu"),
          layers.Dense(6, activation="relu"),
          layers.Dense(6, activation="relu"),
          layers.Dense(8, activation="relu"),
          layers.Dense(input_dim, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def autoencoder_anomaly_detection(input_dim):
    input_layer = Input(shape=(input_dim, ))
    encoder = Dense(8, activation="relu")(input_layer)
    encoder = Dense(6, activation="relu")(encoder)
    encoder = Dense(4, activation="relu")(encoder)
    encoder = Dense(2, activation="relu")(encoder)
    decoder = Dense(4, activation="relu")(encoder)
    decoder = Dense(6, activation="relu")(decoder)
    decoder = Dense(input_dim, activation="sigmoid")(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    return autoencoder


class AnomalyDetectorAutoencoder_No_Q(Model):
    def __init__(self, input_dim):
        super(AnomalyDetectorAutoencoder_No_Q, self).__init__()
        self.encoder = tf.keras.Sequential([
          layers.Dense(5, activation="relu", input_shape=(input_dim,)),
          layers.Dense(4, activation="relu"),
          layers.Dense(4, activation="relu"),
          layers.Dense(3, activation="relu"),
          layers.Dense(2, activation="relu")])
        self.decoder = tf.keras.Sequential([
          layers.Dense(3, activation="relu"),
          layers.Dense(4, activation="relu"),
          layers.Dense(4, activation="relu"),
          layers.Dense(5, activation="relu"),
          layers.Dense(input_dim, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def autoencoder_lstm(dim1, dim2):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(
        units=64,
        input_shape=(dim1, dim2)
    ))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.RepeatVector(n=dim1))
    model.add(keras.layers.LSTM(units=64, return_sequences=True))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Dense(units=dim2)
        )
    )
    return model


def autoencoder_model_1(dim1, dim2):
    input_layer = keras.Input(shape=(dim1, dim2))
    # Encoder
    encoder = layers.LSTM(8, activation="relu", return_sequences=True)(input_layer)
    encoder = layers.LSTM(6, activation="relu", return_sequences=True)(encoder)
    encoder = layers.LSTM(4, activation="relu", return_sequences=True)(encoder)
    # bottleneck
    encoder = layers.LSTM(2, activation="relu", return_sequences=True)(encoder)
    # Decoder
    decoder = layers.LSTM(4, activation="relu", return_sequences=True)(encoder)
    decoder = layers.LSTM(6, activation="relu", return_sequences=True)(decoder)
    decoder = layers.LSTM(8, activation="relu", return_sequences=True)(decoder)
    decoder = Dense(dim1, activation="sigmoid")(decoder)
    model = Model(inputs=input_layer, outputs=decoder)
    return model


def autoencoder_model_2(dim1, dim2):
    input_layer = keras.Input(shape=(dim1, dim2))
    # Encoder
    encoder = layers.LSTM(8, activation="relu", return_sequences=True)(input_layer)
    encoder = layers.LSTM(6, activation="relu", return_sequences=True)(encoder)
    encoder = layers.LSTM(4, activation="relu", return_sequences=True)(encoder)
    encoder = layers.LSTM(2, activation="relu", return_sequences=True)(encoder)
    # bottleneck
    encoder = layers.LSTM(1, activation="relu", return_sequences=True)(encoder)
    # Decoder
    decoder = layers.LSTM(2, activation="relu", return_sequences=True)(encoder)
    decoder = layers.LSTM(4, activation="relu", return_sequences=True)(decoder)
    decoder = layers.LSTM(6, activation="relu", return_sequences=True)(decoder)
    decoder = layers.LSTM(8, activation="relu", return_sequences=True)(decoder)
    decoder = Dense(dim1, activation="sigmoid")(decoder)
    model = Model(inputs=input_layer, outputs=decoder)
    return model


def autoencoder_model(dim1, dim2):
    input_layer = keras.Input(shape=(dim1, dim2))
    # Encoder
    encoder = layers.LSTM(8, activation="relu", return_sequences=True)(input_layer)
    encoder = layers.LSTM(8, activation="relu", return_sequences=True)(encoder)
    encoder = layers.LSTM(6, activation="relu", return_sequences=True)(encoder)
    encoder = layers.LSTM(4, activation="relu", return_sequences=True)(encoder)
    # bottleneck
    encoder = layers.LSTM(2, activation="relu", return_sequences=True)(encoder)
    # Decoder
    decoder = layers.LSTM(4, activation="relu", return_sequences=True)(encoder)
    decoder = layers.LSTM(6, activation="relu", return_sequences=True)(decoder)
    decoder = layers.LSTM(8, activation="relu", return_sequences=True)(decoder)
    decoder = layers.LSTM(8, activation="relu", return_sequences=True)(decoder)
    decoder = Dense(dim1, activation="sigmoid")(decoder)
    model = Model(inputs=input_layer, outputs=decoder)
    return model


def autoencoder_model_3(dim1, dim2):
    input_layer = keras.Input(shape=(dim1, dim2))
    # Encoder
    encoder = layers.LSTM(6, activation="relu", return_sequences=True)(input_layer)
    encoder = layers.LSTM(4, activation="relu", return_sequences=True)(encoder)
    encoder = layers.LSTM(2, activation="relu", return_sequences=True)(encoder)
    # bottleneck
    encoder = layers.LSTM(1, activation="relu", return_sequences=True)(encoder)
    # Decoder
    decoder = layers.LSTM(2, activation="relu", return_sequences=True)(encoder)
    decoder = layers.LSTM(4, activation="relu", return_sequences=True)(decoder)
    decoder = layers.LSTM(6, activation="relu", return_sequences=True)(decoder)
    decoder = Dense(dim1, activation="sigmoid")(decoder)
    model = Model(inputs=input_layer, outputs=decoder)
    return model


def autoencoder_model_4(dim1, dim2):
    input_layer = keras.Input(shape=(dim1, dim2))
    # Encoder
    encoder = layers.LSTM(6, activation="relu", return_sequences=True)(input_layer)
    encoder = layers.LSTM(4, activation="relu", return_sequences=True)(encoder)
    # bottleneck
    encoder = layers.LSTM(2, activation="relu", return_sequences=True)(encoder)
    # Decoder
    decoder = layers.LSTM(4, activation="relu", return_sequences=True)(encoder)
    decoder = layers.LSTM(6, activation="relu", return_sequences=True)(decoder)
    decoder = Dense(dim1, activation="sigmoid")(decoder)
    model = Model(inputs=input_layer, outputs=decoder)
    return model