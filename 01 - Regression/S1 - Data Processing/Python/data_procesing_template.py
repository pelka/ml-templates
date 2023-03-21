# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 11:29:21 2022

@author: pelka
"""

#Plantilla de Pre Procesado de datos

#Como importar las librerias
import pandas as pd

#Importar el data set
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3]. values


#Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


#Escalado de variables
"""
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

"""


































