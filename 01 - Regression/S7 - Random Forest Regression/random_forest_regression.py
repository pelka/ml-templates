# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 10:10:52 2022

@author: pelka
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar dateset

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
"""
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
"""
# Escalado de variables
"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""
# Ajustar la regresion con el dataset

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(x, y)

# Prediccion de nuestro modelo Random Forest

y_pred = regressor.predict([[6.5]])

# Visualizacion de los resultados del Modelo Random Forest

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = "red")
plt.plot(x_grid, regressor.predict(x_grid), color ="blue")
plt.title("Modelo de Regresion")
plt.xlabel("lblx")
plt.ylabel("lbly")
plt.show()