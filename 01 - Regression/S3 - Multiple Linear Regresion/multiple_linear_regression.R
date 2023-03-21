#Regresion Lineal Multiple
#Plantilla para el Pre Procesamiento de Datos
#Importar el dataset

dataset = read.csv('50_Startups.csv')

#Codifying categorical variables
dataset$State = factor(dataset$State,
                         levels = c("New York","California","Florida"),
                         labels = c(1, 2, 3))


#Dividir los datos en conjunto de entrenamiento y conjunto de test
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

#Ajustar el modelo de Regresion Lineal Multiple con el conjunto de entrenamiento
regression = lm(formula = Profit ~ .,
                data = training_set)

#Construir un modelo optimo con la eiliminacion hacia atras

backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
  }
  return(regressor)
}


SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]

#Predecir los resultados con el conjunto de testing
y_pred = predict(backwardElimination(training_set, SL), newdata = testing_set)
