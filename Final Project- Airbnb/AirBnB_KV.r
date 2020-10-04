install.packages(ISLR)
install.packages("tidyr")
install.packages("tidyverse")
library(ISLR)
library("tidyr")
library(tidyverse)

mydata <- read.csv('Data.csv')

mydata = na.omit(mydata)

air_bnb <- mydata[-c(1)]

air_bnb

air_bnb <- air_bnb[-c(1:6,8,10,11,13:17,19:23,26,28:34,36,37,40,41,43:53,59:61)]

air_bnb$AvgNightlyPrice = as.numeric(gsub("[\\$,]", "", air_bnb$AvgNightlyPrice))

air_bnb$AvgNightlyAdjusted_price = as.numeric(gsub("[\\$,]", "", air_bnb$AvgNightlyAdjusted_price))

air_bnb$MinNightlyprice = as.numeric(gsub("[\\$,]", "", air_bnb$MinNightlyprice))

air_bnb$MinNightlyAdjusted_price = as.numeric(gsub("[\\$,]", "", air_bnb$MinNightlyAdjusted_price))

air_bnb$MaxNightlyprice = as.numeric(gsub("[\\$,]", "", air_bnb$MaxNightlyprice ))

air_bnb$MaxNightlyAdjusted_price = as.numeric(gsub("[\\$,]", "", air_bnb$MaxNightlyAdjusted_price))

air_bnb = na.omit(air_bnb)

air_bnb

row_names <- row.names(air_bnb)
row_names

names(air_bnb)

apply(air_bnb, 2, mean)
apply(air_bnb, 2, var)

pc.out=prcomp(air_bnb[,-4], scale. = TRUE, center = TRUE)
summary(pc.out)# First two principal components explain approx 50% variance for 16 variables which is substantial
names(pc.out)
print(pc.out)

pc.out$sdev #standard deviation of each principal components
pc.out$center# sample mean for the original variable
pc.out$scale# standard deviation of the original variable
pc.out$rotation# the contribution of each variable to each principal component
pc.out$x# value of the PCA for every sample points


d<- data.frame(pc.out$x)

pc1_max <- which(d$PC1==max(d$PC1))
pc1_max #8791 row which has high pc1

pc1_min <- which(d$PC1==min(d$PC1))
pc1_min #1989 row which has low pc1


pc2_max <- which(d$PC2==max(d$PC2))
pc2_max #22 row which has high pc2

pc2_min <- which(d$PC2==min(d$PC2))
pc2_min #7780 row which has low pc2


b <- data.frame(cor(air_bnb[,-4])) #Correlation between variables
b

c <- data.frame(pc.out$rotation)
#Plot the first two principal componenets
install.packages("factoextra")
library(factoextra)

fviz_pca_var(pc.out, scale=0, repel = TRUE)
fviz_pca_biplot(pc.out, scale=0,geom = "point",col.ind = air_bnb$property_type)
?fviz_pca_biplot()
biplot(pc.out, scale=0)

unique(air_bnb$property_type)

#The signs of the PC's are not important.
pc.out$rotation=-pc.out$rotation
pc.out$x=-pc.out$x
fviz_pca_biplot(pc.out, scale=0,
                geom = "point",col.ind = air_bnb$property_type,legend.title= "Groups",
                title = "PCA - Biplot",repel = TRUE)
#biplot(pc.out, scale=0,xlabs=rep("$", nrow(air_bnb)))
#PC1 captures information about price variables
# and PC2 is more affected by the ratings.

pc.out$sdev
pc.var=pc.out$sdev^2
pc.var
pvc=pc.var/sum(pc.var)
pvc


#Scree plot
plot(pvc, xlab="Principal Component", ylab="Proportion of Variance Explained", ylim=c(0,1),type='b')
plot(cumsum(pvc), xlab="Principal Component", ylab="Cumulative Proportion of Variance Explained", ylim=c(0,1),type='b')

#From the scree plot we found that, we need 6 PC's are needed for 80% variance to be explained.
#But 50% of Variance can be explained only from the first two PC's which is substantial.

