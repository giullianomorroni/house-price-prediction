#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf8

import numpy as np
import pandas as pd
from tf_keras.models import Sequential
from tf_keras.layers import Dropout, Dense, BatchNormalization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tf_keras.models import save_model, load_model


housing = pd.read_csv('../data/housing_ohe.csv')

housing['area'] = np.log(housing['area'])
housing['bedrooms'] = np.log(housing['bedrooms'])
housing['bathrooms'] = np.log(housing['bathrooms'])
housing['stories'] = np.log(housing['stories'])
housing['price'] = np.log(housing['price'])

housing['parking'] = housing['parking'].apply(lambda x: x if x > 0 else 0.1)
housing['parking'] = np.log(housing['parking'])

# X = housing[['area','bedrooms','bathrooms',
#             'stories','parking','prefarea_ohe',
#             'furnishingstatus_ohe','guestroom_ohe','basement_ohe',
#             'hotwaterheating_ohe','airconditioning_ohe']]

housing = housing.dropna()

X = housing[['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'prefarea_ohe','airconditioning_ohe']]

y = housing['price']

print(X.head(4))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

input_shape = [X_train.shape[1]]

model = Sequential([
    Dense(units=128, activation='relu', input_shape=input_shape),
    Dropout(rate=0.1),
    Dense(units=64, activation='relu'),
    Dropout(rate=0.1),
    Dense(units=1)
])

model.compile(
    optimizer='adam',
    loss='mae'
)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=128, epochs=200, verbose=False)

history_df = pd.DataFrame(history.history)
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))
plt.plot(history_df.loc[:, ['val_loss']], label='val_loss')
plt.plot(history_df.loc[:, ['loss']], label='loss')
plt.legend()
plt.show()

# #save model
# save_model(model, 'model.h5')
# # load model
# model = load_model('model.h5')
# # summarize model.
# model.summary()

for i in range(1,20):
    features = X_test.iloc[[i]]
    real_price = y_test.iloc[[i]]
    real_price = int(np.exp(real_price.values[0]))

    predict_price = model.predict(features)
    predict_price = int(np.exp(predict_price[0][0]))

    percent = (predict_price*100)/real_price

    print('real_price / predict_price (diff)', real_price, '/' ,predict_price, '(', int(percent-100), '%)')
