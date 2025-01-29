#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf8

import pandas as pd
from sklearn.preprocessing import LabelEncoder

housing = pd.read_csv('../data/housing.csv')
encoder = LabelEncoder()

housing['guestroom_ohe'] = encoder.fit_transform(housing['guestroom'])
housing['basement_ohe'] = encoder.fit_transform(housing['basement'])
housing['hotwaterheating_ohe'] = encoder.fit_transform(housing['hotwaterheating'])
housing['airconditioning_ohe'] = encoder.fit_transform(housing['airconditioning'])
housing['prefarea_ohe'] = encoder.fit_transform(housing['prefarea'])
housing['furnishingstatus_ohe'] = encoder.fit_transform(housing['furnishingstatus'])

housing.to_csv('../data/housing_ohe.csv', index=False)
