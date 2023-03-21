# Clustering con K-Means

# Importar los datos

dataset = read.csv("Mall_Customers.csv")
x = dataset[, 4:5]

# Metodo del codo

set.seed(0)

wcss = vector()
for (i in 1:10){
  wcss[i] <- sum(kmeans(x, i)$withinss)
}

plot(1:10, wcss, type = 'b', main = "Metodo del codo",
     xlab = "Numero de clusters (k)", ylab = "WCSS(k)")


# Aplicar el algoritmo de K-Means con k optimo

set.seed(0)
kmeans <- kmeans(x, 5, iter.max = 300, nstart = 10)

# Visualizacion del grafico

library(cluster)

clusplot(x,
         kmeans$cluster,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 4,
         plotchar = FALSE,
         span = TRUE,
         main = "Clustering de clientes",
         xlab = "Ingresos anuales",
         ylab = "Puntuacion (1-100)")

