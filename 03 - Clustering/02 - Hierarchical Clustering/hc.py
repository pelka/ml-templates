# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:04:13 2022

@author: pelka
"""

# Clustering jerarquico

# Importar las librerias de trabajo

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar los datos del centro comercial

dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:,[3,4]].values

# Utilizar el dendrograma para encontrar el nuemero optimo de clusters

import scipy.cluster.hierarchy as sch
dendrograma = sch.dendrogram(sch.linkage(x, method = "ward", ))
plt.title("Dendrograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia Euclidea")
plt.show(dendrograma, block = True)

# Ajustar el clustering jerarquico a nuestro conjunto de datos

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = "ward")
y_hc = hc.fit_predict(x)

# Visualizacion de los clusters

plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 10, c="red", label = "Cautos")
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 10, c="blue", label = "Estandar")
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 10, c="green", label = "Objetivo")
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 10, c="cyan", label = "Descuidados")
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 10, c="magenta", label = "Conservadores")

plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales")
plt.ylabel("Puntuacion de gastos (1 - 100)")
plt.legend()
plt.show()  