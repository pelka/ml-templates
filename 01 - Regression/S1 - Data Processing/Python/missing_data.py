# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 11:58:33 2022

@author: pelka
"""

#Template Pre Processing data - Missing data

import numpy as np
import pandas as pd

#Import dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3]. values


#Processor NA
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, 
                        strategy='mean', 
                        fill_value=None, 
                        verbose=0, 
                        copy=True, 
                        add_indicator=False)
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])