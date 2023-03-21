# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 11:22:10 2022

@author: pelka
"""

# Part 1 - Preprocessing data

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

# Escalado de variables
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Part 2 - Build RNA
# Import libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

# Inicializar la RNA
classifier = Sequential()

# Añadir las capas de entrada y primera capa de resultados

classifier.add(Dense(units = 6, kernel_initializer = "uniform", 
                     activation = "relu", input_dim = 11))

# Añadir la segunda capa oculta
classifier.add(Dense(units = 6, kernel_initializer = "uniform", 
                     activation = "relu"))

# Añadir la capa de salida
classifier.add(Dense(units = 1, kernel_initializer = "uniform", 
                     activation = "sigmoid"))

# Compilar la RNA
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Ajustamos la RNA al Conjunto de entrenamiento
classifier.fit(x_train, y_train, batch_size=10, epochs=100)

# Part 3 - Evaluate model and calculate final predictions

# Prediccion de los resultados con el conjunto de testing

y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# Elaborar una matriz de confusion
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)