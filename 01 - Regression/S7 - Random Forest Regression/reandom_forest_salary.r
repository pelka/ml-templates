# Plantilla de Regresion

# Import dataset

dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]

# Dividir los datos en conjunto de entrenamiento y conjunto de test
# 
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# testing_set = subset(dataset, split == FALSE)

# Ajustar modelo de Regresion con el mismo conjunto de datos

# Crear nuestra variable de Random Forest

# install.packages("randomForest")

library(randomForest)
set.seed(1234)
regression = randomForest(x = dataset[1], 
                           y = dataset$Salary,
                           ntree = 300)


# Prediccion de los resultados --------------------------------------------

y_pred = predict(regression, newdata = data.frame(Level = 6.5))



# Visualizacion del modelo de Random Forest -----------------------------------

library(ggplot2)

x_grid = seq(min(dataset$Level), max(dataset$Level), 0.001)

ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary), 
             color = "red") +
  geom_line(aes(x = x_grid , y = predict(regression, 
                                         newdata = data.frame(Level = x_grid))),
            color = "blue") +
  ggtitle("Random Forest") +
  xlab("") +
  ylab("")
