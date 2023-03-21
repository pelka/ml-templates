# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:01:04 2022

@author: pelka
"""

#XGBoost
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar dateset

dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Codificar datos categoricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

lblEncoder_x1 = LabelEncoder()
x[:, 1] = lblEncoder_x1.fit_transform(x[:,1])

lblEncoder_x2 = LabelEncoder()
x[:, 2] = lblEncoder_x2.fit_transform(x[:,2])

cTransformer_x = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],
                                 remainder='passthrough')
x = np.array(cTransformer_x.fit_transform(x), dtype=float)
x = x[:, 1:]

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Ajustar el modelo xgboost al modelo
from xgboost import XGBClassifier

classifier = XGBClassifier()
