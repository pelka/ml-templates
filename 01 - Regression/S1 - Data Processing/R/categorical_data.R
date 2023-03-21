#Template Pre Processing data - Categorical data
#Import dataset 

dataset = read.csv('data.csv')

#Codifying categorical variables
dataset$Country = factor(dataset$Country,
                         levels = c("France","Spain","Germany"),
                         labels = c(1, 2, 3))

dataset$Purchased = factor(dataset$Purchased,
                           levels = c("No", "Yes"),
                           labels = c(0,1))