# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 16:34:48 2022

@author: pelka
"""

# Natural Language Processing

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)

# Data cleaning
import re 
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 

corpus = []

for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Create Bag of Words
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Ajustar el modelo en el conjunto de entrenamiento
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(x_train, y_train)

# Prediccion de los resultados con el conjunto de testing

y_pred = classifier.predict(x_test)

# Elaborar una matriz de confusion
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)