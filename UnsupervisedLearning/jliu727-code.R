#setwd("Documents/OMSCS/2018sp/CS7641_ML/HW3")
wine<-read.csv(file="winequality-red.csv",  sep=';', header = TRUE)
wine$quality <- factor(wine$quality)
###1.	Run the clustering algorithms on the data sets and describe what you see.
## Wine Quality
#Normalize the data
wine_inputs <-wine[,1:11]
wine_temp<-scale(wine_inputs)
wine[,1:11]<-wine_temp
wine_label<-wine[,12]

#k-means
wss<-c()
times<-c()
for (i in 2:15) {
  wss[i] <- 0
  times[i] <- 0
  for (j in 1:5) {
    ptm <- proc.time()
    wss[i] <- wss[i] + sum(kmeans(wine, i, nstart = 20)$withinss)
    predictTime<-(proc.time() - ptm)[1]
    times[i] <- predictTime + times[i]
  }
  wss[i] = wss[i] / 5
  times[i] = times[i] / 5
}
plot(wss, main="Number of Clusters-Wine", xlab="# of clusters", ylab="within groups sum of squares")
plot(times, main="k-Means Time-Wine", xlab="# of clusters", ylab="time")
wineCluster <- kmeans(wine, 6, nstart = 20)

#EM
wine<-read.csv(file="winequality-red.csv",  sep=';', header = TRUE)
library(mclust)
wineBIC<-mclustBIC(wine)
plot(wineBIC)
#Plot the times
times<-c()
for (i in 2:11) {
  ptm <- proc.time()
  emWineCluster<-Mclust(wine, i)
  times[i] <-(proc.time() - ptm)[1]
}
plot(times, main="Wine EM Timing", xlab="Number of Clusters", ylab="Time")
emWine<-Mclust(wine, 6)
plot(emWine)

##College Ranking
cr<-read.csv(file="cr_new.csv")
#Normalize the data
cr_inputs <-cr[,1:7]
cr_temp<-scale(cr_inputs)
cr[,1:7]<-cr_temp
cr_score<-cr[,8]

#k-means
wss<-c()
times<-c()
for (i in 2:15) {
  wss[i] <- 0
  times[i] <- 0
  for (j in 1:5) {
    ptm <- proc.time()
    wss[i] <- wss[i] + sum(kmeans(cr, i, nstart = 20)$withinss)
    predictTime<-(proc.time() - ptm)[1]
    times[i] <- predictTime + times[i]
  }
  wss[i] = wss[i] / 5
  times[i] = times[i] / 5
}
plot(wss, main="Number of Clusters-CollegeRank", xlab="# of clusters", ylab="within groups sum of squares")
plot(times, main="k-Means Time-CollegeRank", xlab="# of clusters", ylab="time")
crCluster <- kmeans(cr, 6, nstart = 20)

#EM
library(mclust)
crBIC<-mclustBIC(cr)
plot(crBIC)
#Plot the times
times<-c()
for (i in 2:11) {
  ptm <- proc.time()
  emCr<-Mclust(cr, i)
  times[i] <-(proc.time() - ptm)[1]
}
plot(times, main="CollegeRank EM Timing", xlab="Number of Clusters", ylab="Time")
emCr<-Mclust(cr, 6)
plot(emCr)

###2. Apply the dimensionality reduction algorithms to the two datasets and describe what you see.
##PCA
#code from https://github.com/zygmuntz/wine-quality/blob/master/pca_red.r
#Wine Quality
# number of elements
number = length(as.matrix(wine)) / length(wine)

# pca analysis
ptm <- proc.time()
pcx <- prcomp( wine_inputs, scale = TRUE )
predictTime<-(proc.time() - ptm)
print(predictTime)
biplot( pcx, xlabs = rep( '.', number ))

# principal components
bar_colors = c( 'red', 'red', rep( 'gray',10 ))
plot( pcx, col = bar_colors )

#Eigenvalues
ev <- pcx$sdev^2
plot(ev)

variance_plot<-function(dims_red_data, title) {
  #std_dev <- dims_red_data$sdev
  std_dev <- apply(dims_red_data, 2, sd)
  pr_var <- std_dev^2
  pr_var
  prop_varex <- pr_var/sum(pr_var)
  prop_varex
  plot(prop_varex, xlab = "Component",      
       ylab = "Proportion of Variance Explained",
       type = "b", main=title)
  print(prop_varex)
}

variance_plot(pcx$x, "PCA Wine Variance Explained")

#College Ranking
# number of elements
number = length(as.matrix(cr_inputs)) / length(cr_inputs)

# pca analysis
ptm <- proc.time()
cr_pcx <- prcomp( cr_inputs, scale = TRUE )
predictTime<-(proc.time() - ptm)
print(predictTime)
biplot( cr_pcx, xlabs = rep( '.', number ))

# principal components
bar_colors = c( 'red', 'red', rep( 'gray',10 ))
plot( cr_pcx, col = bar_colors )

#ICA
library(fastICA)
library(e1071)
library(ggplot2)
#Wine Data
ptm <- proc.time()
ica <- fastICA(wine_inputs, 11)
predictTime<-(proc.time() - ptm)
print(predictTime)

icaS <- data.frame(ica$S)

icaSkurt <- data.frame(apply(ica$S,2,kurtosis))
icaSkurt <- cbind(icaSkurt,seq(1:11))

icaSkurt[,1] <- abs(icaSkurt[,1] - 3)

names(icaSkurt) = c("kurtosis","component")
ggplot(icaSkurt,aes(x = component, y = kurtosis))+geom_bar(stat="identity") 

#College Ranking Data
ptm <- proc.time()
cr_ica <- fastICA(cr_inputs, 7)
predictTime<-(proc.time() - ptm)
print(predictTime)

cr_icaS <- data.frame(cr_ica$S)

cr_icaSkurt <- data.frame(apply(cr_ica$S,2,kurtosis))
cr_icaSkurt <- cbind(cr_icaSkurt,seq(1:7))

cr_icaSkurt[,1] <- abs(cr_icaSkurt[,1] - 3)

names(cr_icaSkurt) = c("kurtosis","component")
ggplot(cr_icaSkurt,aes(x = component, y = kurtosis))+geom_bar(stat="identity") 

source("random_projection_gauss.R")
library(dplyr)
library(reshape)

# Randomized Projection
# code from https://github.com/chappers/CS7641-Machine-Learning/blob/master/Unsupervised%20Learning/R/random_projection_gauss.R
# Wine
wq_nl = wine[,1:11]
wq_rca <- Map(function(x) {
  gaussian_random_projection(wine[,1:11], 6)
}, 1:100)
# get the ones which immitate the result best.

wqrcadiff <- Map(function(x) {
  sum((wq_nl - (x$RP %*% MASS::ginv(x$R)))^2)
}, wq_rca) %>% melt


bestrca <- wqrcadiff %>% arrange(value) %>% head(1)
names(bestrca) <- c("value", "k")
wqrca <- cbind(as.data.frame(wq_rca[[bestrca$k]]$RP), quality=wine$quality)

#Reconstruction matrix
k = 6
p = ncol(wine)
R <<- matrix(data = rnorm(k*p),
             nrow = k,
             ncol = p)

# College Ranking
cr_wq_nl = cr[,1:7]
cr_wq_rca <- Map(function(x) {
  gaussian_random_projection(cr[,1:7], 6)
}, 1:100)
# get the ones which immitate the result best.

cr_wqrcadiff <- Map(function(x) {
  sum((cr_wq_nl - (x$RP %*% MASS::ginv(x$R)))^2)
}, cr_wq_rca) %>% melt


cr_bestrca <- cr_wqrcadiff %>% arrange(value) %>% head(1)
names(cr_bestrca) <- c("value", "k")
cr_wqrca <- cbind(as.data.frame(cr_wq_rca[[cr_bestrca$k]]$RP), score=cr$score)

#Reconstruction matrix
k = 6
p = ncol(cr)
R <<- matrix(data = rnorm(k*p),
             nrow = k,
             ncol = p)

### Factor Analysis
#code from https://github.com/nirave/unsupervised-learning-nfl-wine/blob/master/main.R
factan <-factanal(wine_inputs, 4)
a<-rowSums(factan$loadings[,1] * wine)
b<-rowSums(factan$loadings[,2] * wine)
c<-rowSums(factan$loadings[,3] * wine)
d<-rowSums(factan$loadings[,4] * wine)

factan_dataset <- cbind(a, b, c, d)

cr_factan <-factanal(cr_inputs, 1)
a<-rowSums(cr_factan$loadings[,1] * cr_inputs)

cr_factan_dataset <- cbind(a)
plot(nfl_factan_dataset, type="n")

library(nFactors)
library(MASS)
ev <- eigen(cor(wine)) # get eigenvalues
ap <- parallel(subject=nrow(wine),var=ncol(wine), rep=100,cent=.05)

nS <- nScree(x=ev$values, aparallel=ap$eigen$qevpea)
plotnScree(nS) 

ev <- eigen(cor(cr_inputs)) # get eigenvalues
ap <- parallel(subject=nrow(cr_inputs),var=ncol(cr_inputs), rep=100,cent=.05)

nS <- nScree(x=ev$values, aparallel=ap$eigen$qevpea)
plotnScree(nS) 

###3.	Reproduce your clustering experiments, but on the data after you've run dimensionality reduction on it.
# adjust code from https://github.com/nirave/unsupervised-learning-nfl-wine/blob/master/main.R
runCluster<-function(data, dataset, results, numClusters, title) {
  wss<-c()
  for (i in 2:15) {
    wss[i] <- 0
    for (j in 1:5) {
      wss[i] <- wss[i] + sum(kmeans(data, i, nstart = 20)$withinss)
    }
    wss[i] = wss[i] / 5
  }
  
  plot(wss)
  
  
  ptm <- proc.time()
  crCluster2 <- kmeans(data, numClusters, nstart = 20)
  predictTime<-(proc.time() - ptm)
  print(predictTime)
  
  crosstab <- table(results, crCluster2$cluster)
  agreement <- randIndex(crosstab)
  print(agreement)
  
  plotcluster(dataset, crCluster2$cluster)
  
  #EM after reduction
  
  ptm <- proc.time()
  emCr2<-Mclust(data, numClusters)
  predictTime<-(proc.time() - ptm)[1]
  print(predictTime)
  
  crosstab <- table(results, emCr2$classification)
  agreement <- randIndex(crosstab)
  print(agreement)
  
  #plotcluster(dataset, emCr2$classification)
}

pcxPCA4d <- cbind(pcx$x[,c(1,2,3,4)],wine[,12])
pcxPCA3d_cr <- cbind(cr_pcx$x[,c(1,2,3)],cr[,12])
runCluster(pcxPCA4d, wine, wine$quality, 6, "Wine PCA")
runCluster(pcxPCA3d_cr, cr, cr$score, 6, "CR PCA")

icsSclasses <- cbind(icaS,wine[,12])
bestICA <- order(icaSkurt$kurtosis)[c(8,9,10,11)]
pcxICA3d <- icsSclasses[,bestICA]
icsSclasses_cr <- cbind(cr_icaS,cr[,8])
bestICA_cr <- order(cr_icaSkurt$kurtosis)[c(4,5,6,7)]
pcxICA3d_cr <- icsSclasses_cr[,bestICA_cr]
runCluster(pcxICA3d, wine, wine$quality, 6, "Wine PCA")
runCluster(pcxICS3d_cr, cr, cr$score, 6, "CR PCA")

runCluster(wqrca, wine, wine$quality, 6, "Wine Random Projection")
runCluster(cr_wqrca, cr, cr$score, 6, "CR Random Projection")

runCluster(factan_dataset, wine, wine$quality, 6, "Wine Factor Analysis")
runCluster(cr_factan_dataset,  cr, cr$score, 6, "CR Factor Analysis")

### 4.	Apply the dimensionality reduction algorithms to one of your datasets from assignment #1 
# (if you've reused the datasets from assignment #1 to do experiments 1-3 above then you've already done this) 
# and rerun your neural network learner on the newly projected data.


#Separate data into training and validation set
cr_new = cr
cr_new$score[cr_new$score <=45] = 0
cr_new$score[cr_new$score >45] = 1
cr_t <- cr_new[1:1100,]
cr_v <- cr_new[1101:2200,]

library(nnet)

cat="score"
ideal <-class.ind(cr_t[[cat]])
nntraining<-subset(cr_t, select = -c(score))

#NN
library(heuristica)
ptm <- proc.time()
nueral_net<-nnet(nntraining, ideal, size=30, softmax=TRUE)
predictions <- predict(nueral_net, cr_v, type="class")
predictTime<-(proc.time() - ptm)
print(predictTime)
predictions = strtoi(predictions, base = 0L)
accuracy = predictions - cr_v['score']
perc = sum(accuracy == 0)/1100

#PCA + NN
range01 <- function(x){(x-min(x))/(max(x)-min(x))}
crnormal<-range01(cr)

pcx_norm <- prcomp(crnormal)
pcx_norm_3d <- cbind(pcx_norm$x[,c(1,2,3)],cr[,7])

cr_nueralnetwork<-function(data, ideal, validation, last) {
  cat = "score"
  new_nntraining<-data[1:1100,]
  new_nntraining<-new_nntraining[,1:last]
  ptm <- proc.time()
  nueral_net<-nnet(new_nntraining, ideal, size=30, softmax=TRUE)
  predictTime<-(proc.time() - ptm)
  print(predictTime)
  predictions <- predict(nueral_net, validation, type="class")
  predictions = strtoi(predictions, base = 0L)
  accuracy = predictions - cr_v['score']
  perc = sum(accuracy == 0)/1100
  print(perc)
}

cr_nueralnetwork(pcx_norm_3d, ideal, cr_v, 3)

#ICA + NN 
ptm <- proc.time()
cr_ica <- fastICA(cr_inputs, 7)
predictTime<-(proc.time() - ptm)
print(predictTime)
cr_icaS <- data.frame(cr_ica$S)
cr_icaSkurt <- data.frame(apply(cr_ica$S,2,kurtosis))
cr_icaSkurt <- cbind(cr_icaSkurt,seq(1:7))
cr_icaSkurt[,1] <- abs(cr_icaSkurt[,1] - 3)
names(cr_icaSkurt) = c("kurtosis","component")
icsSclasses_cr <- cbind(cr_icaS,cr[,8])
bestICA_cr <- order(cr_icaSkurt$kurtosis)[c(4,5,6,7)]
pcxICA3d_cr <- icsSclasses_cr[,bestICA_cr]
cr_nueralnetwork(pcxICA3d_cr, ideal, cr_v, 3)

#Random Projection + NN
cr_nueralnetwork(cr_wqrca, ideal, cr_v, 6)


#Factor Analysis + NN
cr_factan <-factanal(crnormal[1:7], 1)
a<-rowSums(cr_factan$loadings[,1] * cr)
cr_factan_dataset <- cbind(a,b)
cr_nueralnetwork(cr_factan_dataset, ideal, cr_v, 1)

### 5. Apply the clustering algorithms to the same dataset to which you just applied the dimensionality reduction algorithms 
# (you've probably already done this), treating the clusters as if they were new features. 
# In other words, treat the clustering algorithms as if they were dimensionality reduction algorithms. 
# Again, rerun your neural network learner on the newly projected data.

#k-means + PCA
range01 <- function(x){(x-min(x))/(max(x)-min(x))}
crnormal<-range01(cr_new)
pcx_norm <- prcomp(crnormal[1:1100, ])
pcx_norm_3d <- cbind(pcx_norm$x[,c(1,2,3,4)],crnormal[1:1100 ,12])

data<-pcx_norm_3d
numClusters<-6
crClusterNN <- kmeans(data, numClusters, nstart = 20)
new_cr <- cbind(cr_new[1:1100,], crClusterNN$cluster)
new_training <- new_cr
new_validation <- cr_new[1101:2200,]
new_nntraining<-subset(new_training, select = -c(score))
nueral_net<-nnet(new_nntraining, ideal, size=30, softmax=TRUE)
predictTime<-(proc.time() - ptm)
predictions <- predict(nueral_net, new_validation, type="class")
predictions = strtoi(predictions, base = 0L)
accuracy = predictions - new_validation['score']
perc = sum(accuracy == 0)/1100

#EM + PCA
emCrNN<-Mclust(data, 6)
new_cr <- cbind(cr_new[1:1100, ], emCrNN$classification)
new_training <- new_cr
new_validation <- cr_new[1101:2200,]
new_nntraining<-subset(new_training, select = -c(score))
nueral_net<-nnet(new_nntraining, ideal, size=30, softmax=TRUE)
predictTime<-(proc.time() - ptm)
predictions <- predict(nueral_net, new_validation, type="class")
predictions = strtoi(predictions, base = 0L)
accuracy = predictions - new_validation['score']
perc = sum(accuracy == 0)/1100

#k-means
range01 <- function(x){(x-min(x))/(max(x)-min(x))}
crnormal<-range01(cr_new)
data<-crnormal[1:1100, ]
numClusters<-6
crClusterNN <- kmeans(data, numClusters, nstart = 20)
new_cr <- cbind(cr_new[1:1100], crClusterNN$cluster)
new_training <- new_cr
new_validation <- cr_new[1101:2200,]
new_nntraining<-subset(new_training, select = -c(score))
nueral_net<-nnet(new_nntraining, ideal, size=30, softmax=TRUE)
predictTime<-(proc.time() - ptm)
predictions <- predict(nueral_net, new_validation, type="class")
predictions = strtoi(predictions, base = 0L)
accuracy = predictions - new_validation['score']
perc = sum(accuracy == 0)/1100

#EM
emCrNN<-Mclust(data, 6)
new_cr <- cbind(cr_new[1:1100, ], emCrNN$classification)
new_training <- new_cr
new_validation <- cr_new[1101:2200,]
new_nntraining<-subset(new_training, select = -c(score))
nueral_net<-nnet(new_nntraining, ideal, size=30, softmax=TRUE)
predictTime<-(proc.time() - ptm)
predictions <- predict(nueral_net, new_validation, type="class")
predictions = strtoi(predictions, base = 0L)
accuracy = predictions - new_validation['score']
perc = sum(accuracy == 0)/1100
