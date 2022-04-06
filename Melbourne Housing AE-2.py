import keras.models
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
data_colmns = dataframe
data_colmns = np.nan_to_num(data_colmns)
ix = [i for i in range(data_colmns.shape[1])]
data1 = data_colmns[:, ix]
data_colmns = data1[0]

# Reshape the data with the column names, so we can change the data types of each columns individualy
dataValues = dataframe.values
data = pd.DataFrame(dataValues, columns=data_colmns)
data = data.drop([0, 0])
# Get the columns that have categories in them
cols = ['Suburb', 'Address', 'Type', 'Method', 'SellerG', 'Date', 'CouncilArea', 'Regionname']
cols2 = ['Rooms', 'Price', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea',
         'YearBuilt', 'Lattitude', 'Longtitude', 'Propertycount']


# This is copied from https://python.tutorialink.com/how-to-reverse-label-encoder-from-sklearn-for-multiple-columns/.
# Essentially encodes multiple columns, since LabelEncoder() only works on single columns. And we need this to decode the labels later
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


np.set_printoptions(threshold=sys.maxsize)

# encoding happens here
multi = MultiColumnLabelEncoder(columns=cols)

data_na=data[data.isna().any(axis=1)]

data_clean = data.dropna()

data_corrected = data

data_corrected[cols] = data_corrected[cols].fillna('0')
data_corrected[cols2]=data_corrected[cols2].fillna(0)
data = multi.fit_transform(data_corrected)

# get the data that is full of nulls
#print(data_na.values)
# get the clean data and train on it


data_clean=multi.fit_transform(data_corrected)
# np.set_printoptions(threshold=sys.maxsize)

# replace the nulls with 0
data_filled = data

# reshape the data and get the maximum of each column

data_filled = data_filled.values

data_filled = data_filled.astype(float)
ix1 = [i for i in range(data_filled.shape[1])]
maximum = data_filled.max(axis=0)

# this reparametrizes the data from 0 to 1
data_clean = data_clean.values
data_clean = data_clean.astype(float)
ix2 = [i for i in range(data_clean.shape[1])]
data_clean = data_clean[:, ix2]

data_clean = data_clean / maximum
#print(data_clean)

"""
# this splits the data to test and train
test_size = .15

train, test = train_test_split(data_clean, test_size=test_size)
x_train, y_train = train_test_split(train, test_size=test_size, random_state=42)
x_test, y_test = train_test_split(test, test_size=test_size, random_state=42)

input = layers.Input(shape=(21))

dropout_rate = 0.20

# network setup

#x = layers.Dense(21, activation='relu')(input)
#x = layers.Dropout(dropout_rate)(x)
x = layers.Dense(84, activation='relu')(input)
x = layers.Dropout(dropout_rate)(x)
x = layers.Dense(84, activation='relu')(x)
x = layers.Dropout(dropout_rate)(x)
x = layers.Dense(21, activation='tanh')(x)

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

#loss='categorical_crossentropy'
loss = 'mean_squared_error'

opt = keras.optimizers.Adam(learning_rate=0.0001)

autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss=loss)
autoencoder.summary()
autoencoder.fit(
    x=train,
    y=train,
    epochs=500,
    batch_size=100,
    shuffle=True,
    validation_data=(test, test),
    callbacks=tensorboard,
)

# save model
autoencoder.save("autoencoer")


"""

# load model
autoencoder=keras.models.load_model("autoencoer")


np.set_printoptions(threshold=sys.maxsize)
print(data_na.values)
# replace nulls with 0
data_na=data_na.fillna(0)

data_na_values=data_na.values

# encode the data for later
data_na_inv=multi.inverse_transform(data)

# prepare the data to be fit in the trained model
data_na_values=data_na_values.astype(float)
ix3 = [i for i in range(data_na_values.shape[1])]
data_na_values = data_na_values[:, ix3]
data_na_values=data_na_values/maximum

# get data types
types = data.dtypes


#print(types)
prediction=autoencoder.predict(data_na_values)

data_na_check=data_na_values

# get prediction error
check=100*(data_na_check-prediction)/data_na_check

np.set_printoptions(suppress=True)
#print(prediction)


prediction_error=prediction
data_na_error=data_na
prediction=prediction*maximum

prediction=pd.DataFrame(prediction,columns=data_colmns)

prediction=prediction.astype(types)

#print(data)


prediction_inv=multi.inverse_transform(prediction)
#print(prediction_inv)
#print(prediction_inv.shape)

writer = pd.ExcelWriter('prediction.xlsx', engine='xlsxwriter')


prediction_inv.to_excel(writer,sheet_name='Sheet_name_1')
data_na_inv.to_excel(writer,sheet_name='Sheet_name_2')
writer.save()


print(check)

#print(data_na)
#print(prediction)

#weight = autoencoder.get_weights()
#np.savetxt('weight.csv' , weight , fmt='%s', delimiter=',')
weights = autoencoder.layers[1].get_weights()[0]
weights = np.absolute(weights)
weights = np.average(weights,axis=1)
print (weights)
print(data_colmns)
X = weights
Y = data_colmns
Z = [x for _,x in sorted(zip(X,Y))]
Z2= [x for _,x in sorted(zip(X,X))]
print(Z)
print(Z2)

