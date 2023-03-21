# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 18:00:40 2022

@author: Pelka
"""

# Import libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar dateset

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Dividir el data set en conjunto de entrenamiento y conjunto de testing

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Escalado de variables
"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""

# Ajustar la regresion con el dataset


# Prediccion de nuestro modelo

y_pred = regression.predict()

# Visualizacion de los resultados del Modelo Polinomico

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = "red")
plt.plot(x_grid, regression.predict(), color ="blue")
plt.title("Modelo de Regresion")
plt.xlabel("lblx")
plt.ylabel("lbly")
plt.show()