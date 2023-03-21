library(arules)
library(arulesViz)

# data -------------------------------------------------------------------
path <- "~/Downloads/P14-Part5-Association-Rule-Learning/Section 28 - Apriori/"
trans <- read.transactions(
  file = paste0(path, "R/Market_Basket_Optimisation.csv"),
  sep = ",",
  rm.duplicates = TRUE
)

# apriori algoirthm ------------------------------------------------------
rules <- apriori(
  data = trans,
  parameter = list(support = 0.004, confidence = 0.2)
)

# visualizations ---------------------------------------------------------
plot(rules, method = "graph", engine = "htmlwidget")
