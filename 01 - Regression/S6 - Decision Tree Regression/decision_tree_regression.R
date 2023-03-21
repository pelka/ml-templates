# Decision tree regression

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

# Crear nuestra variable de regresion

install.packages("rpart")
library(rpart)

regression = rpart(formula = Salary ~ .,
                   data = dataset,
                   control = rpart.control(minsplit = 1))


# Prediccion de los resultados --------------------------------------------

y_pred_lin = predict(regression, newdata = data.frame(Level = 6.5))



# Visualizacion del modelo de regresion -----------------------------------

library(ggplot2)

x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)

ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary), 
             color = "red") +
  geom_line(aes(x = dataset$Level , y = predict(regression, 
                                         newdata = data.frame(Level = dataset$Level))),
            color = "blue") +
  ggtitle("Prediction with decision tree regression") +
  xlab("Employee level") +
  ylab("Salary $")