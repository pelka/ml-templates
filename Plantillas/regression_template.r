# Plantilla de Regresion

# Import dataset

dataset = read.csv('S4 - Polynomial Regression/Position_Salaries.csv')
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




# Prediccion de los resultados --------------------------------------------

y_pred_lin = predict(lin_reg, newdata = data.frame(Level = 6.5))



# Visualizacion del modelo de regresion -----------------------------------

library(ggplot2)

x_grid = seq(min(dataset$value), max(dataset$values), 0.1)

ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary), 
             color = "red") +
  geom_line(aes(x = x_grid , y = predict(regression, 
                                         newdata = data.frame(value = x_grid))),
            color = "blue") +
  ggtitle("Prediccion") +
  xlab("") +
  ylab("")




