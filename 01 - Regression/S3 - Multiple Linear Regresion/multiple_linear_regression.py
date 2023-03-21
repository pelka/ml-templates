
# Regresion lineal multiple

#Como importar las librerias
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importar el data set
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4]. values

#Codificar datos categoricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labele_X = LabelEncoder()
X[:, 3] = labele_X.fit_transform(X[:,3])
ct_X = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],
                       remainder='passthrough')
X = np.array(ct_X.fit_transform(X), dtype=float)

#Evitar la trampa de las variables dummy

X = X[:, 1:]

#Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Ajustar el modelo de regrresion lineal multiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train, y_train)

#Prediccion de los resultados en el conjunto de testing

y_pred = regression.predict(X_test)

#Construir el modelo optimo de RIM utilixanfo la Elimionacion hacia atras
import statsmodels.api as sm

# Regresion lineal automatica
def backwardElimination(x, sl):    
    numVars = len(x[0])    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        if maxVar > sl:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = np.delete(x, j, 1)    
    regressor_OLS.summary()    
    return x 

SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
