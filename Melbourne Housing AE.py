from pandas import read_csv

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import sys
from numpy import delete
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from keras.utils.np_utils import to_categorical
from keras.losses import MeanSquaredError

from time import time
from tensorflow.python.keras.callbacks import TensorBoard

# load dataset
dataframe = read_csv('melb_data.csv', header=None, na_values='?')

"""
for i in range(dataframe.shape[1]):
    # count number of rows with missing values
    n_miss = dataframe[[i]].isnull().sum()
    perc = n_miss / dataframe.shape[0] * 100
    print('> %d, Missing: %d (%.1f%%)' % (i, n_miss, perc))
"""

# Get the data types from the first row of the set
data_types = dataframe
data_types = np.nan_to_num(data_types)
ix = [i for i in range(data_types.shape[1])]
data1 = data_types[:, ix]
data_types = data1[0]

# Reshape the data with the column names, so we can change the data types of each columns individualy
dataValues = dataframe.values
data = pd.DataFrame(dataValues, columns=data_types)

data = data.drop([0, 0])
data_na = data

# Get the columns that have categories in them
cols = ['Suburb', 'Address', 'Type', 'Method', 'SellerG', 'Date', 'CouncilArea', 'Regionname']


class MultiColumnLabelEncoder:

    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def fit(self, X, y=None):
        self.encoders = {}
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            self.encoders[col] = LabelEncoder().fit(X[col])
        return self

    def transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].transform(X[col])
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].inverse_transform(X[col])
        return output


# Encode the columns and add 1. 0 is kept for missing values.
# data[cols] = data[cols].apply(LabelEncoder().fit_transform)

multi=MultiColumnLabelEncoder(columns=cols)
data=multi.fit_transform(data)
data[cols] = data[cols] + 1

data = data.dropna()

data = data.values

# np.set_printoptions(threshold=sys.maxsize)

# reshape the data
ix1 = [i for i in range(data.shape[1])]
data = data[:, ix1]

# turn into floats
data = data.astype(float)

# reparametrize
max = data.max(axis=0)
data = data / max

print(max)

# replace empty with 0
# data = np.nan_to_num(data)


# print(data.shape)
# print(data)

test_size = .1

train, test = train_test_split(data, test_size=test_size)

x_train, y_train = train_test_split(train, test_size=test_size, random_state=42)
x_test, y_test = train_test_split(test, test_size=test_size, random_state=42)

input = layers.Input(shape=(21))

dropout_rate = 0.2

x = layers.Dense(21, activation='relu')(input)
x = layers.Dropout(dropout_rate)(x)
x = layers.Dense(16, activation='relu')(x)
x = layers.Dropout(dropout_rate)(x)
x = layers.Dense(16, activation='relu')(x)
x = layers.Dropout(dropout_rate)(x)
x = layers.Dense(21, activation='sigmoid')(x)

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# loss='categorical_crossentropy'
loss = 'mean_squared_error'

autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss=loss)
autoencoder.summary()

autoencoder.fit(
    x=train,
    y=train,
    epochs=1,
    batch_size=3500,
    shuffle=True,
    validation_data=(test, test),
    callbacks=tensorboard,
)
#predictions = autoencoder.predict(test) * max

data_na_encoded=multi.fit_transform(data_na)

print(data_na_encoded)

data_na_encoded[cols] = data_na_encoded[cols] + 1

data_na_encoded = data_na_encoded[data_na_encoded.isna().any(axis=1)]
print(data_na_encoded)

"""

data_na_encoded = data_na_encoded.values
# reshape the data
ix1 = [i for i in range(data_na_encoded.shape[1])]
data_na_encoded = data_na_encoded[:, ix1]

data_na_encoded = np.nan_to_num(data_na_encoded)

# turn into floats
data_na_encoded = data_na_encoded.astype(float)

# reparametrize
data_na_encoded = data_na_encoded * max

predicted_na_encoded = autoencoder.predict(data_na_encoded)

data_na_encoded = pd.DataFrame(data_na_encoded, columns=[data_types])

data_na_encoded[cols] = data_na_encoded[cols] - 1

data_na=multi.inverse_transform(data_na_encoded)

data_na = data_na.values

print(data_na)
"""