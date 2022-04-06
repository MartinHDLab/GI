from pandas import read_csv

import pandas as pd
import numpy as np
from numpy import delete
import tensorflow as tf
import matplotlib.pyplot as plt

import seaborn as sns
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize




# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv'
dataframe = read_csv(url, header=None, na_values='?')
"""
for i in range(dataframe.shape[1]):
    # count number of rows with missing values
    n_miss = dataframe[[i]].isnull().sum()
    perc = n_miss / dataframe.shape[0] * 100
    print('> %d, Missing: %d (%.1f%%)' % (i, n_miss, perc))
"""

data = dataframe.values


data = np.delete(data,2,1)



data = np.nan_to_num(data)
max = data.max(axis=0)
data = data/max

print(pd.DataFrame(data))


ix = [i for i in range(data.shape[1])]

data = data[:, ix]

print(data.shape)







test_size=.1

train, test = train_test_split(data, test_size=test_size)




#x_train,y_train = train_test_split(train, test_size=test_size, random_state=42)
#x_test,y_test = train_test_split(test, test_size=test_size, random_state=42)





input = layers.Input(shape=(27))


x=layers.Dense(27,activation='relu')(input)
#x=layers.Dense(64,activation='sigmoid')(x)
x=layers.Dense(16,activation='relu')(x)
x=layers.Dense(8,activation='relu')(x)
x=layers.Dense(4,activation='sigmoid')(x)
x=layers.Dense(8,activation='relu')(x)
x=layers.Dense(16,activation='relu')(x)
#x=layers.Dense(64,activation='sigmoid')(x)
x=layers.Dense(27,activation='sigmoid')(x)

autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="categorical_crossentropy")
autoencoder.summary()

autoencoder.fit(
    x=train,
    y=train,
    epochs=800,
    batch_size=80,
    shuffle=True,
    validation_data=(test, test),
)
predictions = autoencoder.predict(test)*max

#autoencoder.save_weights('weights.h5', overwrite=True, save_format='h5')

weight = autoencoder.get_weights()
np.savetxt('weight.csv' , weight , fmt='%s', delimiter=',')

#print(pd.DataFrame(test)*max)
#print(pd.DataFrame(predictions))

difference = ((test-predictions)/max)*100
print(pd.DataFrame(difference))


