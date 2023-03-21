# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 11:57:13 2022

@author: pelka
"""
#Template Pre Processing data - Categorical data

import numpy as np
import pandas as pd

#Import dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3]. values

#Codificar datos categoricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labele_X = LabelEncoder()
X[:, 0] = labele_X.fit_transform(X[:,0])
ct_X = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],
                       remainder='passthrough')
X = np.array(ct_X.fit_transform(X), dtype=float)
label_y = LabelEncoder()
y = label_y.fit_transform(y)