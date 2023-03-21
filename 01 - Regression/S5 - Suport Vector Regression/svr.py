# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:42:14 2022

@author: pelka
"""

#SVR

# Importar librerias

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values.reshape(-1,1)

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
"""
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
"""
# Escalado de variables

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# Ajustar la regresion con el dataset
from sklearn.svm import SVR
regression = SVR(kernel = 'rbf')
regression.fit(x, y) 


# Prediccion de nuestro modelo con SVR

y_pred = sc_y.inverse_transform(regression.predict(sc_x.transform([[6.5]])).reshape(-1,1))

                      
# Visualizacion de los resultados SVR

x_plot = sc_x.inverse_transform(x[:9,])
y_plot = sc_y.inverse_transform(y[:9,])

x_grid =  np.arange(min(x_plot), max(x_plot), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x_plot, y_plot, color = "red")
plt.plot(x_grid, sc_y.inverse_transform(regression.predict(sc_x.transform(x_grid)).reshape(-1,1)), color ="blue")
plt.title("Modelo de Regresion (SVR)")
plt.xlabel("lblx")
plt.ylabel("lbly")
plt.show()