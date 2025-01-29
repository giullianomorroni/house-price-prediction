#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn


df = pd.read_csv('../data/housing_ohe.csv')

df = df[['price', 'area','bedrooms','bathrooms',
        'stories','parking','prefarea_ohe',
        'furnishingstatus_ohe','guestroom_ohe','basement_ohe',
        'hotwaterheating_ohe','airconditioning_ohe']]

matrix = df.corr(method ='pearson')

mask = np.triu(np.ones_like(matrix))
heatmap = seaborn.heatmap(matrix, mask=mask, annot=True, cmap='BrBG')
plt.show()
