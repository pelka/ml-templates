# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 09:27:10 2022

@author: pelka
"""

# K-Means

# Import librarys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import data

dataset = pd.read_csv("Mall_Customers.csv")

x = dataset.iloc[:,[3,4]].values

# Metodo del codo para averiguar el numero optimo de clusters

from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)
plt.title("Metodo del codo")
plt.xlabel("Numero de clusters")
plt.ylabel("WCSS(k)")
plt.show()

# Aplicar el metodo de K-Means para segmentar el dataset

kmeans = KMeans(n_clusters = 5, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)

# Visualizacion de los clusters

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 10, c="red", label = "Cautos")
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 10, c="blue", label = "Estandar")
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 10, c="green", label = "Objetivo")
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 10, c="cyan", label = "Descuidados")
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 10, c="magenta", label = "Conservadores")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 150, c="yellow", label = " Baricentros")

plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales")
plt.ylabel("Puntuacion de gastos (1 - 100)")
plt.legend()
plt.show()
