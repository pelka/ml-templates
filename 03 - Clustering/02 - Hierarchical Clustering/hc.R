# Clustering jerarquico

# Importar los datos del centro comercial

dataset = read.csv("Mall_Customers.csv")
x = dataset[, 4:5]

#Utilizar el dendrograma para encontrar el numero optimo de  clusters

dendrogram = hclust(dist(x, method = "euclidean"),
                    method = "ward.D")

plot(dendrogram,
     main = "Dendrograma",
     xlab = "Clientes del centro comercial",
     ylab = "Distancia Euclidea")

# Ajustar el clustering jerarquico a nuestro dataset

hc = hclust(dist(x, method = "euclidean"),
                    method = "ward.D")
y_hc = cutree(hc, k = 5)

# Visualizar los clusters

library(cluster)

clusplot(x,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 4,
         plotchar = FALSE,
         span = TRUE,
         main = "Clustering de clientes",
         xlab = "Ingresos anuales",
         ylab = "Puntuacion (1-100)")