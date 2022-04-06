from sklearn.preprocessing import LabelEncoder

data=[1,2,2,6]

le=LabelEncoder().fit(data)

data=(LabelEncoder().fit(data)).transform(data)

print(data)

L=le.inverse_transform(data)

print(L)