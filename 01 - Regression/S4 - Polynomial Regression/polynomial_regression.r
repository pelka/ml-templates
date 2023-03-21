# Regresion polinomica

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

# Ajustar modelo de Regresion Lineal con el conjunto de datos

lin_reg = lm(formula = Salary ~ ., 
             data = dataset)

# Ajustar modelo de Regresion Polinomica con el mismo conjunto de datos

dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4

poly_reg = lm(formula = Salary ~ .,
              data = dataset)

# Visualizacion del modelo lineal

library(ggplot2)

ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary), 
             color = "red") +
  geom_line(aes(x = dataset$Level , y = predict(lin_reg, newdata = dataset)),
            color = "blue") +
  ggtitle("Prediccion lineal del sueldo en funcion del nivel del empleado") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")

# Visualizacion del modelo Polinomico

x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)

ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary), 
             color = "red") +
  geom_line(aes(x = x_grid , y = predict(poly_reg, newdata = data.frame(Level = x_grid, 
                                                                        Level2 = x_grid^2,
                                                                        Level3 = x_grid^3,
                                                                        Level4 = x_grid^4))),
            color = "blue") +
  ggtitle("Prediccion lineal del sueldo en funcion del nivel del empleado") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")

#Prediccion de nuevos resultados con regresion lineal

y_pred_lin = predict(lin_reg, newdata = data.frame(Level = 6.5))

#Prediccion de nuevos resultados con regresion polinomica

y_pred_poly = predict(poly_reg, newdata = data.frame(Level = 6.6, 
                                                     Level2 = 6.5^2,
                                                     Level3 = 6.5^3,
                                                     Level4 = 6.5^4))
