#SVR
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

# Ajustar SVR de Regresion con el mismo conjunto de datos ---------

# Install.packages("e1071")
library(e1071)

regression = svm( formula = Salary ~ .,
                  data = dataset,
                  type = "eps-regression",
                  kernel = "radial",
                  degree = 3,
                  cost = 9 )

# Crear nuestra variable de regresion




# Result Predicts --------------------------------------------

y_pred_lin = predict(regression, newdata = data.frame(Level = 6.5))



# Regression Model visualization -----------------------------
library(ggplot2)

x_grid = seq(min(dataset$value), max(dataset$values), 0.1)

ggplot() +
  geom_point(aes(x = dataset$Level , y = dataset$Salary),
             color = "red") +
  geom_line(aes(x = dataset$Level, y = predict(regression, 
                                               newdata = data.frame(Level = dataset$Level))),
            color = "blue") +
  ggtitle("Predicción (SVR)") +
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")


