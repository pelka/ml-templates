# Natural Language Processing

# Import dataset
dataset_original = read.delim("Restaurant_Reviews.tsv", quote = '',
                     stringsAsFactors = FALSE)

# Data cleaning
# Consultar el primer elemento del corpus
# as.character(corpus[[1]])
#install.packages('tm')
#install.packages("SnowballC")
library(tm)
library(SnowballC)

corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords(kind = "en"))
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# Create bag of words

dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)

dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

# Codificar la variable de clasificacion como factor

dataset$Liked = factor(dataset$Liked, levels = c(0,1))

# Split dataset in training_set and testing_set
library(caTools)

set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.80)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Ajustar el modelo de regresion con el conjunto de entrenamiento

# install.packages("randomForest")
library(randomForest)

classifier = randomForest(x = training_set[,-692],
                          y = training_set$Liked,
                          ntree = 10)

# Prediccion de los resultados con el conjunto de testing

y_pred = predict(classifier, newdata = testing_set[,-692])

# Crear la matriz de confusion

cm = table(testing_set[,692], y_pred)
