# Apriori

# Data preprocessing
# install.packages("arules")

library(arules)

dataset = read.csv("Market_Basket_Optimisation.csv", header = FALSE)
dataset = read.transactions("Market_Basket_Optimisation.csv",
                            sep = ",",
                            rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Algorithm training with dataset

rules = apriori(data = dataset,
                parameter = list(support = 0.0037, confidence = 0.2))

# Display results

inspect(sort(rules, by = 'lift')[1:10])
