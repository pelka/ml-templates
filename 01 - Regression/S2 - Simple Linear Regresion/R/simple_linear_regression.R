# Simple linear regression

# Import dataset

dataset = read.csv('Salary_Data.csv')


#Dividir los datos en conjunto de entrenamiento y conjunto de test
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)


# Adjust simple linear regression with training set

regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)


# Predict training set results





















