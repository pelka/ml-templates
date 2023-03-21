# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 10:17:28 2022

@author: pelka
"""

#Regresion Lineal Simple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


#Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)


#Escalado de variables
"""
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

"""

# Create simple linear regresion model with training set
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predict test set
y_pred = regression.predict(X_test)

#visualizate training results
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regression.predict(X_train), color="blue")
plt.title("Sueldo vs años de experiencia (Conjunto de entrenamiento)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()

#visualizate test results
plt.scatter(X_test, y_test, color="green")
plt.plot(X_train, regression.predict(X_train), color="blue")
plt.title("Sueldo vs años de experiencia (Conjunto de test)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()





































