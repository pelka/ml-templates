# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 07:21:06 2022

@author: pelka
"""

# Apriori

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset 
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header = None)
transactions = []

for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0,20)])
    
# Entrenar el algoritmo de apriori
from py.apyori import apriori

rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, 
                min_lift = 3, min_lenght = 2) 

# Visualizacion de los resultados

results = list(rules) 

