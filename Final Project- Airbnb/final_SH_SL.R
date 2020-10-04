# Group 8 - Sarah Hayes, Sean Lei, Kausik Valeti
# STAT 515 - Applied Statistics and Visualization for Analytics
# Final Project - Predicting Airbnb Price

####################################################################################
# Import required packages
library(tidyverse)
library(MASS)
library(ggplot2)
library(randomForest)
library(dplyr)
library(devtools)
library(ggbiplot)
library(e1071)
library("cluster")
library("factoextra")
library(gridExtra)
library(MLmetrics)

# Read in dataset
#setwd("C:/Users/sbcar/Desktop/School/STAT 515/Final")
airbnb_data = read.csv(
  file="DataCleansed.csv",
  header=T, as.is=TRUE)
head(airbnb_data)

#Divide the data into training and testing sets
set.seed(57)
traindat = sample(1:nrow(airbnb_data),0.7*nrow(airbnb_data))
testdat = airbnb_data[-traindat,]
####################################################################################
#Exploratory Analysis

ggplot(airbnb_data, aes(x=rev_rating2)) + geom_histogram()
ggplot(airbnb_data, aes(x=rev_clean2)) + geom_histogram()
ggplot(airbnb_data, aes(x=rev_val2)) + geom_histogram()
ggplot(airbnb_data, aes(x=rev_comm2)) + geom_histogram()
ggplot(airbnb_data, aes(x=rev_loc2)) + geom_histogram()

#Looking at histograms for price - they are all skewed
ggplot(airbnb_data, aes(x=AvgNightlyPrice)) + geom_histogram()
ggplot(airbnb_data, aes(x=MinNightlyprice)) + geom_histogram()
ggplot(airbnb_data, aes(x=MaxNightlyprice)) + geom_histogram()
ggplot(airbnb_data, aes(x=Cost_room)) + geom_histogram()
ggplot(airbnb_data, aes(x=Cost_person_included)) + geom_histogram()
skewness(airbnb_data$AvgNightlyPrice)

#Looking at histograms for rating - they are also all skewed
ggplot(airbnb_data, aes(x=rev_clean)) + geom_histogram()
ggplot(airbnb_data, aes(x=rev_loc)) + geom_histogram()

ggplot(airbnb_data, aes(x=rev_comm)) + geom_histogram()
ggplot(airbnb_data, aes(x=rev_rating)) + geom_histogram()
ggplot(airbnb_data, aes(x=rev_val)) + geom_histogram()

#rev_rating2<-  as.numeric(airbnb_data$rev_rating^3-min(airbnb_data$rev_rating^3))/(max(airbnb_data$rev_rating^3)-min(airbnb_data$rev_rating^3))
#rev_acc2<-  (airbnb_data$rev_acc^2-min(airbnb_data$rev_acc^2))/(max(airbnb_data$rev_acc^2)-min(airbnb_data$rev_acc^2))
rev_clean2<-  (airbnb_data$rev_clean^2-min(airbnb_data$rev_clean^2))/(max(airbnb_data$rev_clean^2)-min(airbnb_data$rev_clean^2))
rev_val2<-  (airbnb_data$rev_val^2-min(airbnb_data$rev_val^2))/(max(airbnb_data$rev_val^2)-min(airbnb_data$rev_val^2))
rev_check2<-  (airbnb_data$rev_check^2-min(airbnb_data$rev_check^2))/(max(airbnb_data$rev_check^2)-min(airbnb_data$rev_check^2))
rev_comm2<-  (airbnb_data$rev_comm^3-min(airbnb_data$rev_comm^3))/(max(airbnb_data$rev_comm^3)-min(airbnb_data$rev_comm^3))
rev_loc2<-  (airbnb_data$rev_loc^7-min(airbnb_data$rev_loc^7))/(max(airbnb_data$rev_loc^7)-min(airbnb_data$rev_loc^7))

ggplot(airbnb_data, aes(x=rev_clean2)) + geom_histogram()
ggplot(airbnb_data, aes(x=rev_loc2)) + geom_histogram()
skewness(airbnb_data$rev_clean)
skewness(rev_clean2)
skewness(airbnb_data$rev_loc)
skewness(rev_loc2)

####################################################################################
#Pre-processing

#transform attibute and remove outliers
AvgNightlyPrice1=sqrt(airbnb_data$AvgNightlyPrice)
AvgNightlyPrice2=(AvgNightlyPrice1-min(AvgNightlyPrice1))/(max(AvgNightlyPrice1)-min(AvgNightlyPrice1))
ggplot(airbnb_data, aes(x=AvgNightlyPrice2)) + geom_histogram()
subset1=airbnb_data[AvgNightlyPrice2 <= .55,]
ggplot(subset1, aes(x=sqrt(AvgNightlyPrice))) + geom_histogram()

#transform attibute and remove outliers
MinNightlyprice1=sqrt(subset1$MinNightlyprice)
MinNightlyprice2=(MinNightlyprice1-min(MinNightlyprice1))/(max(MinNightlyprice1)-min(MinNightlyprice1))
ggplot(subset1, aes(x=MinNightlyprice2)) + geom_histogram()
subset2=subset1[MinNightlyprice2 <= .75,]
ggplot(subset2, aes(x=sqrt(MinNightlyprice))) + geom_histogram()
nrow(subset2)

#transform attibute and remove outliers
MaxNightlyprice1=sqrt(subset2$MaxNightlyprice)
MaxNightlyprice2=(MaxNightlyprice1-min(MaxNightlyprice1))/(max(MaxNightlyprice1)-min(MaxNightlyprice1))
ggplot(subset2, aes(x=MaxNightlyprice2)) + geom_histogram()
subset3=subset2[MaxNightlyprice2 <= .55,]
ggplot(subset3, aes(x=sqrt(MaxNightlyprice))) + geom_histogram()
nrow(subset3)

#create subset of data to use for prices for linear regression
airbnb_data_price=subset3
AvgNightlyPrice1=sqrt(airbnb_data_price$AvgNightlyPrice)
AvgNightlyPrice2=(AvgNightlyPrice1-min(AvgNightlyPrice1))/(max(AvgNightlyPrice1)-min(AvgNightlyPrice1))
MinNightlyprice1=sqrt(airbnb_data_price$MinNightlyprice)
MinNightlyprice2=(MinNightlyprice1-min(MinNightlyprice1))/(max(MinNightlyprice1)-min(MinNightlyprice1))
MaxNightlyprice1=sqrt(airbnb_data_price$MaxNightlyprice)
MaxNightlyprice2=(MaxNightlyprice1-min(MaxNightlyprice1))/(max(MaxNightlyprice1)-min(MaxNightlyprice1))

#transform additional variables
Cost_room1=sqrt(airbnb_data_price$Cost_room)
Cost_room2=(Cost_room1-min(Cost_room1))/(max(Cost_room1)-min(Cost_room1))
Cost_person_included1=sqrt(airbnb_data_price$Cost_person_included)
Cost_person_included2=(Cost_person_included1-min(Cost_person_included1))/(max(Cost_person_included1)-min(Cost_person_included1))

airbnb_data_price$AvgNightlyPrice2=AvgNightlyPrice2
airbnb_data_price$MinNightlyprice2=MinNightlyprice2
airbnb_data_price$MaxNightlyprice2=MaxNightlyprice2
airbnb_data_price$Cost_room2=Cost_room2
airbnb_data_price$Cost_person_included2=Cost_person_included2

#view plots after transformation
ggplot(airbnb_data_price, aes(x=AvgNightlyPrice2)) + geom_histogram()
ggplot(airbnb_data_price, aes(x=MinNightlyprice2)) + geom_histogram()
ggplot(airbnb_data_price, aes(x=MaxNightlyprice2)) + geom_histogram()
ggplot(airbnb_data_price, aes(x=Cost_room2)) + geom_histogram()
ggplot(airbnb_data_price, aes(x=Cost_person_included2)) + geom_histogram()
skewness(airbnb_data$AvgNightlyPrice2)

#location rating
airbnb_data_price$rev_loc3<-(airbnb_data_price$rev_loc^7-min(airbnb_data_price$rev_loc^7))/(max(airbnb_data_price$rev_loc^7)-min(airbnb_data_price$rev_loc^7))
  
#Normalizing all variables
airbnb_data_price$bedrooms2<-  as.numeric(airbnb_data_price$bedrooms-min(airbnb_data_price$bedrooms))/(max(airbnb_data_price$bedrooms)-min(airbnb_data_price$bedrooms))
airbnb_data_price$bathrooms2<-  as.numeric(airbnb_data_price$bathrooms-min(airbnb_data_price$bathrooms))/(max(airbnb_data_price$bathrooms)-min(airbnb_data_price$bathrooms))
airbnb_data_price$cancel_pol2<-  as.numeric(airbnb_data_price$cancel_pol-min(airbnb_data_price$cancel_pol))/(max(airbnb_data_price$cancel_pol)-min(airbnb_data_price$cancel_pol))
airbnb_data_price$room_cat2<-  as.numeric(airbnb_data_price$room_cat-min(airbnb_data_price$room_cat))/(max(airbnb_data_price$room_cat)-min(airbnb_data_price$room_cat))
airbnb_data_price$Family_Friendly2<-  as.numeric(airbnb_data_price$Family_Friendly-min(airbnb_data_price$Family_Friendly))/(max(airbnb_data_price$Family_Friendly)-min(airbnb_data_price$Family_Friendly))
airbnb_data_price$Pets_Allowed2<-  as.numeric(airbnb_data_price$Pets_Allowed-min(airbnb_data_price$Pets_Allowed))/(max(airbnb_data_price$Pets_Allowed)-min(airbnb_data_price$Pets_Allowed))
airbnb_data_price$Smoking_Allowed2<-  as.numeric(airbnb_data_price$Smoking_Allowed-min(airbnb_data_price$Smoking_Allowed))/(max(airbnb_data_price$Smoking_Allowed)-min(airbnb_data_price$Smoking_Allowed))
airbnb_data_price$Events_Allowed2<-  as.numeric(airbnb_data_price$Events_Allowed-min(airbnb_data_price$Events_Allowed))/(max(airbnb_data_price$Events_Allowed)-min(airbnb_data_price$Events_Allowed))
airbnb_data_price$rev_rating2<-  as.numeric(airbnb_data_price$rev_rating-min(airbnb_data_price$rev_rating))/(max(airbnb_data_price$rev_rating)-min(airbnb_data_price$rev_rating))
airbnb_data_price$rev_acc2<-  as.numeric(airbnb_data_price$rev_acc-min(airbnb_data_price$rev_acc))/(max(airbnb_data_price$rev_acc)-min(airbnb_data_price$rev_acc))
airbnb_data_price$rev_clean2<-  as.numeric(airbnb_data_price$rev_clean-min(airbnb_data_price$rev_clean))/(max(airbnb_data_price$rev_clean)-min(airbnb_data_price$rev_clean))
airbnb_data_price$rev_check2<-  as.numeric(airbnb_data_price$rev_check-min(airbnb_data_price$rev_check))/(max(airbnb_data_price$rev_check)-min(airbnb_data_price$rev_check))
airbnb_data_price$rev_comm2<-  as.numeric(airbnb_data_price$rev_comm-min(airbnb_data_price$rev_comm))/(max(airbnb_data_price$rev_comm)-min(airbnb_data_price$rev_comm))
airbnb_data_price$rev_loc2<-  as.numeric(airbnb_data_price$rev_loc-min(airbnb_data_price$rev_loc))/(max(airbnb_data_price$rev_loc)-min(airbnb_data_price$rev_loc))
airbnb_data_price$rev_val2<-  as.numeric(airbnb_data_price$rev_val-min(airbnb_data_price$rev_val))/(max(airbnb_data_price$rev_val)-min(airbnb_data_price$rev_val))
airbnb_data_price$host_responsetime2<-  as.numeric(airbnb_data_price$host_responsetime-min(airbnb_data_price$host_responsetime))/(max(airbnb_data_price$host_responsetime)-min(airbnb_data_price$host_responsetime))
airbnb_data_price$host_response_rate2<-  as.numeric(airbnb_data_price$host_response_rate-min(airbnb_data_price$host_response_rate))/(max(airbnb_data_price$host_response_rate)-min(airbnb_data_price$host_response_rate))
airbnb_data_price$host_is_superhost2<-  as.numeric(airbnb_data_price$host_is_superhost-min(airbnb_data_price$host_is_superhost))/(max(airbnb_data_price$host_is_superhost)-min(airbnb_data_price$host_is_superhost))
airbnb_data_price$host_listings_count2<-  as.numeric(airbnb_data_price$host_listings_count-min(airbnb_data_price$host_listings_count))/(max(airbnb_data_price$host_listings_count)-min(airbnb_data_price$host_listings_count))
airbnb_data_price$accommodates2<-  as.numeric(airbnb_data_price$accommodates-min(airbnb_data_price$accommodates))/(max(airbnb_data_price$accommodates)-min(airbnb_data_price$accommodates))
airbnb_data_price$Basic_Amenities2<-  as.numeric(airbnb_data_price$Basic_Amenities-min(airbnb_data_price$Basic_Amenities))/(max(airbnb_data_price$Basic_Amenities)-min(airbnb_data_price$Basic_Amenities))
airbnb_data_price$deluxe_amenities2<-  as.numeric(airbnb_data_price$deluxe_amenities-min(airbnb_data_price$deluxe_amenities))/(max(airbnb_data_price$deluxe_amenities)-min(airbnb_data_price$deluxe_amenities))
airbnb_data_price$Total_Amenities2<-  as.numeric(airbnb_data_price$Total_Amenities-min(airbnb_data_price$Total_Amenities))/(max(airbnb_data_price$Total_Amenities)-min(airbnb_data_price$Total_Amenities))
airbnb_data_price$security_deposit2<-  as.numeric(airbnb_data_price$security_deposit-min(airbnb_data_price$security_deposit))/(max(airbnb_data_price$security_deposit)-min(airbnb_data_price$security_deposit))
airbnb_data_price$cleaning_fee2<-  as.numeric(airbnb_data_price$cleaning_fee-min(airbnb_data_price$cleaning_fee))/(max(airbnb_data_price$cleaning_fee)-min(airbnb_data_price$cleaning_fee))
airbnb_data_price$number_of_reviews2<-  as.numeric(airbnb_data_price$number_of_reviews-min(airbnb_data_price$number_of_reviews))/(max(airbnb_data_price$number_of_reviews)-min(airbnb_data_price$number_of_reviews))
airbnb_data_price$reviews_per_month2<-  as.numeric(airbnb_data_price$reviews_per_month-min(airbnb_data_price$reviews_per_month))/(max(airbnb_data_price$reviews_per_month)-min(airbnb_data_price$reviews_per_month))
airbnb_data_price$NearbyArson2<-  as.numeric(airbnb_data_price$NearbyArson-min(airbnb_data_price$NearbyArson))/(max(airbnb_data_price$NearbyArson)-min(airbnb_data_price$NearbyArson))
airbnb_data_price$NearbyAssault2<-  as.numeric(airbnb_data_price$NearbyAssault-min(airbnb_data_price$NearbyAssault))/(max(airbnb_data_price$NearbyAssault)-min(airbnb_data_price$NearbyAssault))
airbnb_data_price$NearbyBikeShare2<-  as.numeric(airbnb_data_price$NearbyBikeShare-min(airbnb_data_price$NearbyBikeShare))/(max(airbnb_data_price$NearbyBikeShare)-min(airbnb_data_price$NearbyBikeShare))
airbnb_data_price$NearbyClassASexOffenders2<-  as.numeric(airbnb_data_price$NearbyClassASexOffenders-min(airbnb_data_price$NearbyClassASexOffenders))/(max(airbnb_data_price$NearbyClassASexOffenders)-min(airbnb_data_price$NearbyClassASexOffenders))
airbnb_data_price$NearbyGunShots2<-  as.numeric(airbnb_data_price$NearbyGunShots-min(airbnb_data_price$NearbyGunShots))/(max(airbnb_data_price$NearbyGunShots)-min(airbnb_data_price$NearbyGunShots))
airbnb_data_price$NearbyHomicide2<-  as.numeric(airbnb_data_price$NearbyHomicide-min(airbnb_data_price$NearbyHomicide))/(max(airbnb_data_price$NearbyHomicide)-min(airbnb_data_price$NearbyHomicide))
airbnb_data_price$NearbyMetros2<-  as.numeric(airbnb_data_price$NearbyMetros-min(airbnb_data_price$NearbyMetros))/(max(airbnb_data_price$NearbyMetros)-min(airbnb_data_price$NearbyMetros))
airbnb_data_price$NearbyMuseums2<-  as.numeric(airbnb_data_price$NearbyMuseums-min(airbnb_data_price$NearbyMuseums))/(max(airbnb_data_price$NearbyMuseums)-min(airbnb_data_price$NearbyMuseums))
airbnb_data_price$NearbySexCrime2<-  as.numeric(airbnb_data_price$NearbySexCrime-min(airbnb_data_price$NearbySexCrime))/(max(airbnb_data_price$NearbySexCrime)-min(airbnb_data_price$NearbySexCrime))
airbnb_data_price$NearbySexOffenders2<-  as.numeric(airbnb_data_price$NearbySexOffenders-min(airbnb_data_price$NearbySexOffenders))/(max(airbnb_data_price$NearbySexOffenders)-min(airbnb_data_price$NearbySexOffenders))
airbnb_data_price$Closest_BikeShare2<-  as.numeric(airbnb_data_price$Closest_BikeShare-min(airbnb_data_price$Closest_BikeShare))/(max(airbnb_data_price$Closest_BikeShare)-min(airbnb_data_price$Closest_BikeShare))
airbnb_data_price$Closest_Metro2<-  as.numeric(airbnb_data_price$Closest_Metro-min(airbnb_data_price$Closest_Metro))/(max(airbnb_data_price$Closest_Metro)-min(airbnb_data_price$Closest_Metro))
airbnb_data_price$Closest_Museum2<-  as.numeric(airbnb_data_price$Closest_Museum-min(airbnb_data_price$Closest_Museum))/(max(airbnb_data_price$Closest_Museum)-min(airbnb_data_price$Closest_Museum))
airbnb_data_price$Closest_SexCrime2<-  as.numeric(airbnb_data_price$Closest_SexCrime-min(airbnb_data_price$Closest_SexCrime))/(max(airbnb_data_price$Closest_SexCrime)-min(airbnb_data_price$Closest_SexCrime))
airbnb_data_price$Closest_SexOffender2<-  as.numeric(airbnb_data_price$Closest_SexOffender-min(airbnb_data_price$Closest_SexOffender))/(max(airbnb_data_price$Closest_SexOffender)-min(airbnb_data_price$Closest_SexOffender))
airbnb_data_price$Closest_Arson2<-  as.numeric(airbnb_data_price$Closest_Arson-min(airbnb_data_price$Closest_Arson))/(max(airbnb_data_price$Closest_Arson)-min(airbnb_data_price$Closest_Arson))
airbnb_data_price$Closest_Assault2<-  as.numeric(airbnb_data_price$Closest_Assault-min(airbnb_data_price$Closest_Assault))/(max(airbnb_data_price$Closest_Assault)-min(airbnb_data_price$Closest_Assault))
airbnb_data_price$Closest_GunShot2<-  as.numeric(airbnb_data_price$Closest_GunShot-min(airbnb_data_price$Closest_GunShot))/(max(airbnb_data_price$Closest_GunShot)-min(airbnb_data_price$Closest_GunShot))
airbnb_data_price$Closest_Homicide2<-  as.numeric(airbnb_data_price$Closest_Homicide-min(airbnb_data_price$Closest_Homicide))/(max(airbnb_data_price$Closest_Homicide)-min(airbnb_data_price$Closest_Homicide))
airbnb_data_price$Closest_ClassA_SexOffender2<-  as.numeric(airbnb_data_price$Closest_ClassA_SexOffender-min(airbnb_data_price$Closest_ClassA_SexOffender))/(max(airbnb_data_price$Closest_ClassA_SexOffender)-min(airbnb_data_price$Closest_ClassA_SexOffender))

#Separate new dataset into train and test
traindat_p = sample(1:nrow(airbnb_data_price),0.7*nrow(airbnb_data_price))
testdat_p = airbnb_data_price[-traindat_p,]

#######################################################################################
#Analysis of average nightly price
lm.avgpricenorm<-lm(AvgNightlyPrice2~bedrooms2+bathrooms2+cancel_pol2+room_cat2+Family_Friendly2+Pets_Allowed2+Smoking_Allowed2+Events_Allowed2+rev_rating2+rev_acc2+rev_clean2+rev_check2+rev_comm2+rev_loc2+rev_val2+host_responsetime2+host_response_rate2+host_is_superhost2+host_listings_count2+accommodates2+Basic_Amenities2+deluxe_amenities2+Total_Amenities2+security_deposit2+cleaning_fee2+number_of_reviews2+reviews_per_month2+NearbyArson2+NearbyAssault2+NearbyBikeShare2+NearbyClassASexOffenders2+NearbyGunShots2+NearbyHomicide2+NearbyMetros2+NearbyMuseums2+NearbySexCrime2+NearbySexOffenders2+Closest_BikeShare2+Closest_Metro2+Closest_Museum2+Closest_SexCrime2+Closest_SexOffender2+Closest_Arson2+Closest_Assault2+Closest_GunShot2+Closest_Homicide2+Closest_ClassA_SexOffender2
                ,data=airbnb_data_price[traindat_p,])
summary(lm.avgpricenorm)

stepavgnorm <- stepAIC(lm.avgpricenorm2, direction="both")
stepavgnorm$anova

colsavgnorm=AvgNightlyPrice2 ~ bedrooms2 + bathrooms2 + cancel_pol2 + room_cat2 + 
  Family_Friendly2 + rev_clean2 + rev_loc2 + host_is_superhost2 + 
  accommodates2 + Basic_Amenities2 + deluxe_amenities2 + cleaning_fee2 + 
  reviews_per_month2 + NearbyAssault2 + NearbyBikeShare2 + 
  NearbyMuseums2 + Closest_SexOffender2 + Closest_Arson2 + 
  Closest_GunShot2

lm.avgpricenorm2<-lm(colsavgnorm,data=airbnb_data_price[traindat_p,])
summary(lm.avgpricenorm2)
plot(lm.avgpricenorm2)

predictavgprice=predict.lm(lm.avgpricenorm2,newdata=airbnb_data_price[-traindat_p,])
test_avgprice=airbnb_data_price[-traindat_p,]

#calculate error
MSE=mean((test_avgprice$AvgNightlyPrice2-predictavgprice)^2)

min(airbnb_data_price$AvgNightlyPrice)+MSE*(max(airbnb_data_price$AvgNightlyPrice)-min(airbnb_data_price$AvgNightlyPrice))

predictdollar=(predictavgprice^2*(max(airbnb_data_price$AvgNightlyPrice)-min(airbnb_data_price$AvgNightlyPrice))+min(airbnb_data_price$AvgNightlyPrice))
nightlypricedollar=(test_avgprice$AvgNightlyPrice2^2*(max(airbnb_data_price$AvgNightlyPrice)-min(airbnb_data_price$AvgNightlyPrice))+min(airbnb_data_price$AvgNightlyPrice))

MSE_dollar=mean((nightlypricedollar-predictdollar)^2)

RMSE=sqrt(MSE_dollar)


RMSPE(y_pred=predictavgprice,y_true=airbnb_data_price[-traindat_p,]$AvgNightlyPrice)



#Curious because random forest MSE was higher. Checking what MSE on training data was. Somehow performed better on test than training.
#trainpredict=predict.lm(lm.avgpricenorm2,newdata=airbnb_data_price[traindat_p,])
#train_avgprice=airbnb_data_price[traindat_p,]
#MSEtrain=mean((train_avgprice$AvgNightlyPrice2-trainpredict)^2)

#trnpredictdollar=trainpredict*(max(airbnb_data_price$AvgNightlyPrice)-min(airbnb_data_price$AvgNightlyPrice))+min(airbnb_data_price$AvgNightlyPrice)
#trnnightlypricedollar=train_avgprice$AvgNightlyPrice2*(max(airbnb_data_price$AvgNightlyPrice)-min(airbnb_data_price$AvgNightlyPrice))+min(airbnb_data_price$AvgNightlyPrice)
#MSE_dollartrn=mean((trnnightlypricedollar-trnpredictdollar)^2)

########################################################################################
#Plots if we hadn't normalized the data

colsavg2=AvgNightlyPrice ~ bedrooms + bathrooms + cancel_pol + room_cat + 
  Family_Friendly + rev_rating + rev_clean + rev_loc + 
  rev_val + host_is_superhost + accommodates + Basic_Amenities + 
  deluxe_amenities + cleaning_fee + reviews_per_month + 
  NearbyAssault + NearbyBikeShare + NearbyClassASexOffenders + 
  NearbyMuseums + NearbySexOffenders + Closest_BikeShare + 
  Closest_Metro + Closest_SexOffender + Closest_Arson + 
  Closest_GunShot

lm.avgprice4<-lm(colsavg2,data=airbnb_data[traindat,])
summary(lm.avgprice4)
plot(lm.avgprice4)

#######################################################################################
#PCA

airbnb_data_small=select(airbnb_data_price[traindat_p,],AvgNightlyPrice,bedrooms,accommodates,room_cat,Closest_SexOffender,bedrooms,rev_rating,Closest_GunShot,Closest_Museum,Events_Allowed,bathrooms,rev_clean,host_response_rate,Closest_Metro,Basic_Amenities,rev_loc,Smoking_Allowed,rev_val,deluxe_amenities,Family_Friendly)
airbnb_data_small.pca <- prcomp(airbnb_data_small, center = TRUE,scale. = TRUE)

#airbnb_data_large=select(airbnb_data_price[traindat_p,],AvgNightlyPrice,bathrooms,bedrooms,beds,guests_included,extra_people,cancel_pol,room_cat,Family_Friendly,Pets_Allowed,Smoking_Allowed,Events_Allowed,rev_rating,rev_acc,rev_clean,rev_check,rev_comm,rev_loc,rev_val,host_responsetime,host_response_rate,host_is_superhost,host_listings_count,accommodates,Basic_Amenities,deluxe_amenities,Total_Amenities,security_deposit,cleaning_fee,number_of_reviews,reviews_per_month,NearbyArson,NearbyAssault,NearbyBikeShare,NearbyClassASexOffenders,NearbyGunShots,NearbyHomicide,NearbyMetros,NearbyMuseums,NearbySexCrime,NearbySexOffenders,Closest_BikeShare,Closest_Metro,Closest_Museum,Closest_SexCrime,Closest_SexOffender,Closest_Arson,Closest_Assault,Closest_GunShot,Closest_Homicide,Closest_ClassA_SexOffender,square_feet,len_desc,len_summary,len_rules)
#airbnb_data_large.pca <- prcomp(airbnb_data_large, center = TRUE,scale. = TRUE)

summary(airbnb_data_small.pca)
str(airbnb_data_small.pca)
ggbiplot(airbnb_data_small.pca,ellipse=TRUE,groups=airbnb_data_small$room_cat,varname.size = 5,varname.abbrev = FALSE) 
ggbiplot(airbnb_data_small.pca,ellipse=TRUE,groups=airbnb_data_small$room_cat,choices=c(1,4),varname.size = 5,varname.abbrev = FALSE)
ggbiplot(airbnb_data_small.pca,ellipse=TRUE,groups=airbnb_data_small$room_cat,choices=c(1,16),varname.size = 5,varname.abbrev = FALSE)

ggbiplot(airbnb_data_small.pca,ellipse=TRUE,groups=airbnb_data_small$bedrooms,varname.size = 5,varname.abbrev = FALSE)
ggbiplot(airbnb_data_small.pca,ellipse=TRUE,groups=airbnb_data_small$bathrooms,varname.size = 5,varname.abbrev = FALSE)
ggbiplot(airbnb_data_small.pca,ellipse=TRUE,groups=airbnb_data_small$accommodates,varname.size = 5,varname.abbrev = FALSE)
ggbiplot(airbnb_data_small.pca,ellipse=TRUE,groups=airbnb_data_small$Events_Allowed,varname.size = 5,varname.abbrev = FALSE)
ggbiplot(airbnb_data_small.pca,ellipse=TRUE,groups=airbnb_data_small$Smoking_Allowed,varname.size = 5,varname.abbrev = FALSE)
ggbiplot(airbnb_data_small.pca,ellipse=FALSE,groups=airbnb_data_small$rev_rating,choices=c(1,7),varname.size = 5,varname.abbrev = FALSE)
ggbiplot(airbnb_data_small.pca,ellipse=FALSE,groups=airbnb_data_small$AvgNightlyPrice,choices=c(1,4),varname.size = 5,varname.abbrev = FALSE)

summary(airbnb_data_large.pca)
str(airbnb_data_large.pca)

#ggbiplot(airbnb_data_large.pca,ellipse=TRUE,groups=airbnb_data_large$room_cat,varname.size = 5,varname.abbrev = FALSE) 
#ggbiplot(airbnb_data_large.pca,ellipse=TRUE,groups=airbnb_data_large$room_cat,choices=c(3,4),varname.size = 5,varname.abbrev = FALSE)


##################################################################################################################################################################################
#Begin Sean Lei
##################################################################################################################################################################################

#K-means with k=2
#K-means clustering depends on the initial cluster assignment
#By setting nstart=25, k-means are performed with 25 different 
#initial assignment and report the best result.
km.out = kmeans(airbnb_data, 2, nstart = 25)

# Save cluster numbers - as.factor to differentiate clusters
km.clusters = as.factor(km.out$cluster)
#table(km.clusters,hc.clusters)

#ggplot(airbnb_data, aes(rev_rating, AvgNightlyPrice, color = km.clusters)) + geom_point()
fviz_cluster(km.out, data = airbnb_data)

k2 = kmeans(airbnb_data[,c("cleaning_fee","AvgNightlyPrice")], centers = 2, nstart = 25)
str(k2)

# Visualize clusters
#fviz_cluster(k2, data = airbnb_data[,c(25,54)], geom="point")
fviz_cluster(k2, data = airbnb_data, geom="point", choose.vars = c("cleaning_fee", "AvgNightlyPrice"), stand = FALSE) + theme_bw()
#If there are more than two dimensions (variables) fviz_cluster will 
# perform principal component analysis and plot the data points 
# according to the first two principal components that explain the 
# majority of the variance.

k3 = kmeans(airbnb_data[,c("cleaning_fee","AvgNightlyPrice")], centers = 3, nstart = 25)
k4 = kmeans(airbnb_data[,c("cleaning_fee","AvgNightlyPrice")], centers = 4, nstart = 25)
k5 = kmeans(airbnb_data[,c("cleaning_fee","AvgNightlyPrice")], centers = 5, nstart = 25)

# plots to compare
p1 = fviz_cluster(k2, data = airbnb_data[,c("cleaning_fee","AvgNightlyPrice")],geom = "point") + ggtitle("k = 2")
p2 = fviz_cluster(k3, data = airbnb_data[,c("cleaning_fee","AvgNightlyPrice")],geom = "point") + ggtitle("k = 3")
p3 = fviz_cluster(k4, data = airbnb_data[,c("cleaning_fee","AvgNightlyPrice")],geom = "point") + ggtitle("k = 4")
p4 = fviz_cluster(k5, data = airbnb_data[,c("cleaning_fee","AvgNightlyPrice")],geom = "point") + ggtitle("k = 5")

grid.arrange(p1, p2, p3, p4, nrow = 2)

# Compute total within-cluster sum of square 
wss = function(k) {
  kmeans(airbnb_data, k, nstart = 10 )$tot.withinss
}

# Compute and plot wss for k = 2 to k = 10
k.values = 2:10

# extract wss for 2-10 clusters
wss_values = map_dbl(k.values, wss)

plot(k.values, wss_values,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares",
     main="Sum of Squares vs Cluster Number")

#One way to determine k is to choose the k where you see a bend.


####################################################################################################################################
# Random Forest to predict Airbnb average nightly price using all variables except for prices
# Grow the random forest using 8 randomly chosen variables at each split and 300 individual trees. 

rf.price=randomForest(AvgNightlyPrice~.-AvgNightlyAdjusted_price-id-MaxNightlyprice-MaxNightlyAdjusted_price-MinNightlyprice-MinNightlyAdjusted_price,data=airbnb_data,subset=traindat,mtry=8,importance=TRUE, ntree=300)

# See error rate vs number of trees
plot(rf.price, main='Number of Trees vs Error')

# Determine most important variables for predicting price
importance(rf.price)
varImpPlot(rf.price, main="Random Forest Variable Importance")

# Answer:
# cleaning fee

# c) Predict the average nightly price on the test set based on the fitted random forest model

yhat.rf = predict(rf.price,newdata=testdat)

oob.err=double(19)
test.err=double(19)

#mtry is number of Variables randomly chosen at each split
for(mtry in 1:19) 
{
  rf=randomForest(AvgNightlyPrice~.-id-AvgNightlyAdjusted_price-MaxNightlyprice-MaxNightlyAdjusted_price-MinNightlyprice-MinNightlyAdjusted_price, data = airbnb_data , subset = traindat,mtry=mtry,ntree=300) 
  #Error of all Trees fitted
  oob.err[mtry] = rf$mse[100] 
  
  #Predictions on Test Set for each Tree
  pred = predict(rf,testdat) 
  #Mean Squared Test Error
  test.err[mtry] = with(testdat, mean( (AvgNightlyPrice - pred)^2)) 
  #Print the output to the console
  cat(mtry," ") 
  
}

# Plot test error
matplot(1:mtry , cbind(oob.err,test.err), pch=19 , col=c("red","blue"),type="b",main="Calculated Mean Square Error with Varying Number of Predictors",ylab="Mean Squared Error",xlab="Number of Predictors Considered at each Split")
legend("topright",legend=c("Out of Bag Error","Test Error"),pch=19, col=c("red","blue"))


RMSPE(y_pred=yhat.rf,y_true=testdat$AvgNightlyPrice)

####################################################################################################################################
# Random Forest on normalized data 
 

#rf.price2=randomForest(AvgNightlyPrice2~bedrooms2+bathrooms2+cancel_pol2+room_cat2+Family_Friendly2+Pets_Allowed2+Smoking_Allowed2+Events_Allowed2+rev_rating2+rev_acc2+rev_clean2+rev_check2+rev_comm2+rev_loc2+rev_val2+host_responsetime2+host_response_rate2+host_is_superhost2+host_listings_count2+accommodates2+Basic_Amenities2+deluxe_amenities2+Total_Amenities2+security_deposit2+cleaning_fee2+number_of_reviews2+reviews_per_month2+NearbyArson2+NearbyAssault2+NearbyBikeShare2+NearbyClassASexOffenders2+NearbyGunShots2+NearbyHomicide2+NearbyMetros2+NearbyMuseums2+NearbySexCrime2+NearbySexOffenders2+Closest_BikeShare2+Closest_Metro2+Closest_Museum2+Closest_SexCrime2+Closest_SexOffender2+Closest_Arson2+Closest_Assault2+Closest_GunShot2+Closest_Homicide2+Closest_ClassA_SexOffender2,
#                                           data=airbnb_data_price,subset=traindat_p,mtry=8,importance=TRUE, ntree=300)

rf.price2=randomForest(AvgNightlyPrice~bedrooms+bathrooms+cancel_pol+room_cat+Family_Friendly+Pets_Allowed+Smoking_Allowed+Events_Allowed+rev_rating+rev_acc+rev_clean+rev_check+rev_comm+rev_loc+rev_val+host_responsetime+host_response_rate+host_is_superhost+host_listings_count+accommodates+Basic_Amenities+deluxe_amenities+Total_Amenities+security_deposit+cleaning_fee+number_of_reviews+reviews_per_month+NearbyArson+NearbyAssault+NearbyBikeShare+NearbyClassASexOffenders+NearbyGunShots+NearbyHomicide+NearbyMetros+NearbyMuseums+NearbySexCrime+NearbySexOffenders+Closest_BikeShare+Closest_Metro+Closest_Museum+Closest_SexCrime+Closest_SexOffender+Closest_Arson+Closest_Assault+Closest_GunShot+Closest_Homicide+Closest_ClassA_SexOffender,
                       data=airbnb_data_price,subset=traindat_p,mtry=8,importance=TRUE, ntree=300)


# See error rate vs number of trees
plot(rf.price2, main='Number of Trees vs Error')

# Determine most important variables for predicting price
importance(rf.price2)
varImpPlot(rf.price2, main="Random Forest Variable Importance")

# Answer:
# cleaning fee

# c) Predict the average nightly price on the test set based on the fitted random forest model

yhat.rf2 = predict(rf.price2,newdata=airbnb_data_price[-traindat_p,])

mean((airbnb_data_price[-traindat_p,]$AvgNightlyPrice2-yhat.rf2)^2)


predictdollar2=(yhat.rf2^2*(max(airbnb_data_price$AvgNightlyPrice)-min(airbnb_data_price$AvgNightlyPrice))+min(airbnb_data_price$AvgNightlyPrice))
nightlypricedollar2=(test_avgprice$AvgNightlyPrice2^2*(max(airbnb_data_price$AvgNightlyPrice)-min(airbnb_data_price$AvgNightlyPrice))+min(airbnb_data_price$AvgNightlyPrice))

MSE_dollar2=mean((nightlypricedollar2-predictdollar2)^2)


##################################################################################################################################################################################
##################################################################################################################################################################################
##################################################################################################################################################################################
##################################################################################################################################################################################
############# Miscellaneous other relationships explored below this point, not used in write up


########################################################################################
#Analysis for location rating

lm.loc<-lm(rev_loc3~Cost_room2+Closest_BikeShare2+Closest_Metro2+Closest_SexCrime2+Closest_SexOffender2+Closest_Arson2+Closest_Assault2+
           Closest_GunShot2+Closest_Homicide2+Closest_ClassA_SexOffender2,data=airbnb_data_price[traindat_p,])
summary(lm.loc)
plot(lm.loc)
lm.loc2<-lm(rev_loc3~Cost_room2+NearbyBikeShare2+NearbyMetros2+NearbySexCrime2+NearbySexOffenders2+NearbyArson2+NearbyAssault2+
              NearbyGunShots2+NearbyHomicide2+NearbyClassASexOffenders2,data=airbnb_data_price[traindat_p,])
summary(lm.loc2)
plot(lm.loc2)
lm.locall<-lm(rev_loc3~Cost_room2+NearbyBikeShare2+NearbyMetros2+NearbySexCrime2+NearbySexOffenders2+NearbyArson2+NearbyAssault2+
                NearbyGunShots2+NearbyHomicide2+NearbyClassASexOffenders2+Closest_BikeShare2+Closest_Metro2+Closest_SexCrime2+Closest_SexOffender2+
              Closest_Arson2+Closest_Assault2+Closest_GunShot2+Closest_Homicide2+Closest_ClassA_SexOffender2,data=airbnb_data_price[traindat_p,])
summary(lm.locall)
plot(lm.locall)
steploc <- stepAIC(lm.locall, direction="both")
steploc$anova
cols_loc=rev_loc3~Cost_room2 + NearbyBikeShare2 + NearbyMetros2 + NearbySexCrime2 + 
  NearbySexOffenders2 + NearbyGunShots2 + NearbyHomicide2 + 
  NearbyClassASexOffenders2 + Closest_BikeShare2 + Closest_Metro2 + 
  Closest_SexOffender2   
lm.locfit<-lm(cols_loc,data=airbnb_data_price[traindat_p,])
summary(lm.locfit) 
plot(lm.locfit)

#########################################################################################
#Analysis for Max Nightly Price per property
lm.maxprice<-lm(MaxNightlyprice2~bedrooms+bathrooms+cancel_pol+room_cat+Family_Friendly+Pets_Allowed+Smoking_Allowed+Events_Allowed+rev_rating+rev_acc+rev_clean+rev_check+rev_comm+rev_loc+rev_val+host_responsetime+host_response_rate+host_is_superhost+host_listings_count+accommodates+Basic_Amenities+deluxe_amenities+Total_Amenities+security_deposit+cleaning_fee+number_of_reviews+reviews_per_month+NearbyArson+NearbyAssault+NearbyBikeShare+NearbyClassASexOffenders+NearbyGunShots+NearbyHomicide+NearbyMetros+NearbyMuseums+NearbySexCrime+NearbySexOffenders+Closest_BikeShare+Closest_Metro+Closest_Museum+Closest_SexCrime+Closest_SexOffender+Closest_Arson+Closest_Assault+Closest_GunShot+Closest_Homicide+Closest_ClassA_SexOffender
                ,data=airbnb_data_price[traindat_p,])
stepmax <- stepAIC(lm.maxprice, direction="both")
stepmax$anova
colsmax=MaxNightlyprice2 ~ bedrooms + room_cat + Family_Friendly + Smoking_Allowed + 
  Events_Allowed + rev_rating + rev_clean + rev_comm + rev_loc + 
  rev_val + host_responsetime + host_response_rate + host_is_superhost + 
  accommodates + Basic_Amenities + deluxe_amenities + cleaning_fee + 
  NearbyAssault + NearbyBikeShare + NearbyClassASexOffenders + 
  NearbyGunShots + NearbyMuseums + NearbySexOffenders + Closest_BikeShare + 
  Closest_Metro + Closest_Museum + Closest_SexOffender + Closest_Arson + 
  Closest_GunShot
lm.pricemax2<-lm(colsmax,data=airbnb_data_price[traindat_p,])
summary(lm.pricemax2)
plot(lm.pricemax2)
lm.pricemaxtest<-lm(colsmax,data=airbnb_data_price[-traindat_p,])
summary(lm.pricemaxtest)

########################################################################################
#Analysis for Min Nightly Price per property
lm.minprice<-lm(MinNightlyprice2~bedrooms+bathrooms+cancel_pol+room_cat+Family_Friendly+Pets_Allowed+Smoking_Allowed+Events_Allowed+rev_rating+rev_acc+rev_clean+rev_check+rev_comm+rev_loc+rev_val+host_responsetime+host_response_rate+host_is_superhost+host_listings_count+accommodates+Basic_Amenities+deluxe_amenities+Total_Amenities+security_deposit+cleaning_fee+number_of_reviews+reviews_per_month+NearbyArson+NearbyAssault+NearbyBikeShare+NearbyClassASexOffenders+NearbyGunShots+NearbyHomicide+NearbyMetros+NearbyMuseums+NearbySexCrime+NearbySexOffenders+Closest_BikeShare+Closest_Metro+Closest_Museum+Closest_SexCrime+Closest_SexOffender+Closest_Arson+Closest_Assault+Closest_GunShot+Closest_Homicide+Closest_ClassA_SexOffender
                ,data=airbnb_data_price[traindat_p,])
stepmin <- stepAIC(lm.minprice, direction="both")
stepmin$anova
colsmin=MinNightlyprice2 ~ bedrooms + bathrooms + cancel_pol + room_cat + 
  Family_Friendly + Pets_Allowed + Smoking_Allowed + rev_rating + 
  rev_clean + rev_comm + rev_loc + rev_val + host_responsetime + 
  host_is_superhost + Basic_Amenities + deluxe_amenities + 
  security_deposit + cleaning_fee + reviews_per_month + NearbyArson + 
  NearbyAssault + NearbyBikeShare + NearbyClassASexOffenders + 
  NearbyMetros + NearbyMuseums + NearbySexOffenders + Closest_BikeShare + 
  Closest_Metro + Closest_Museum + Closest_SexOffender + Closest_Arson + 
  Closest_GunShot
lm.minprice2<-lm(colsmin,data=airbnb_data_price[traindat_p,])
summary(lm.minprice2)
plot(lm.minprice2)
lm.pricemintest<-lm(colsmin,data=airbnb_data_price[-traindat_p,])
summary(lm.pricemintest)

########################################################################################
#Analysis for Avg Nightly Price per Bedroom

lm.roomprice<-lm(Cost_room2~cancel_pol+room_cat+Family_Friendly+Pets_Allowed+Smoking_Allowed+Events_Allowed+rev_rating+rev_acc+rev_clean+rev_check+rev_comm+rev_loc+rev_val+host_responsetime+host_response_rate+host_is_superhost+host_listings_count+accommodates+Basic_Amenities+deluxe_amenities+Total_Amenities+security_deposit+cleaning_fee+number_of_reviews+reviews_per_month+NearbyArson+NearbyAssault+NearbyBikeShare+NearbyClassASexOffenders+NearbyGunShots+NearbyHomicide+NearbyMetros+NearbyMuseums+NearbySexCrime+NearbySexOffenders+Closest_BikeShare+Closest_Metro+Closest_Museum+Closest_SexCrime+Closest_SexOffender+Closest_Arson+Closest_Assault+Closest_GunShot+Closest_Homicide+Closest_ClassA_SexOffender
                 ,data=airbnb_data_price[traindat_p,])
steproom <- stepAIC(lm.roomprice, direction="both")
steproom$anova
colsroom=Cost_room2 ~ room_cat + Family_Friendly + Pets_Allowed +  Events_Allowed + rev_rating + rev_clean + rev_comm + rev_loc + 
  rev_val + host_is_superhost + accommodates + Basic_Amenities + deluxe_amenities + number_of_reviews + reviews_per_month + 
  NearbyAssault + NearbyBikeShare + Closest_Metro + Closest_Museum + Closest_SexOffender + Closest_Arson + Closest_GunShot + Closest_Homicide
lm.roomprice2<-lm(colsroom,data=airbnb_data_price[traindat_p,])
summary(lm.roomprice2)
plot(lm.roomprice2)
lm.priceroomtest<-lm(colsroom,data=airbnb_data_price[-traindat_p,])
summary(lm.priceroomtest)

########################################################################################
#Analysis for Avg Nightly Price per Person

lm.personprice<-lm(Cost_person_included2~bedrooms2+bathrooms2+cancel_pol2+room_cat2+Family_Friendly2+Pets_Allowed2+Smoking_Allowed2+Events_Allowed2+rev_rating2+rev_acc2+rev_clean2+rev_check2+rev_comm2+rev_loc2+rev_val2+host_responsetime2+host_response_rate2+host_is_superhost2+host_listings_count2+accommodates2+Basic_Amenities2+deluxe_amenities2+Total_Amenities2+security_deposit2+cleaning_fee2+number_of_reviews2+reviews_per_month2+NearbyArson2+NearbyAssault2+NearbyBikeShare2+NearbyClassASexOffenders2+NearbyGunShots2+NearbyHomicide2+NearbyMetros2+NearbyMuseums2+NearbySexCrime2+NearbySexOffenders2+Closest_BikeShare2+Closest_Metro2+Closest_Museum2+Closest_SexCrime2+Closest_SexOffender2+Closest_Arson2+Closest_Assault2+Closest_GunShot2+Closest_Homicide2+Closest_ClassA_SexOffender2
                   ,data=airbnb_data_price[traindat_p,])
stepperson <- stepAIC(lm.personprice, direction="both")
stepperson$anova
colsperson=Cost_person_included2 ~ bedrooms2 + bathrooms2 + cancel_pol2 + 
  room_cat2 + rev_rating2 + rev_clean2 + rev_comm2 + rev_loc2 + 
  rev_val2 + host_responsetime2 + accommodates2 + Basic_Amenities2 + 
  deluxe_amenities2 + Total_Amenities2 + security_deposit2 + 
  reviews_per_month2 + NearbyAssault2 + NearbyBikeShare2 + 
  NearbyClassASexOffenders2 + NearbyMetros2 + NearbyMuseums2 + 
  NearbySexOffenders2 + Closest_Metro2 + Closest_Museum2 + 
  Closest_Arson2 + Closest_Assault2 + Closest_GunShot2
lm.personprice2<-lm(colsperson,data=airbnb_data_price[traindat_p,])
summary(lm.personprice2)
plot(lm.personprice2)
lm.personpricetest<-lm(colsperson,data=airbnb_data_price[-traindat_p,])
summary(lm.personpricetest)

########################################################################################
#Analysis for Average Nightly Price per property, no normalization
lm.avgprice<-lm(AvgNightlyPrice2~bedrooms+bathrooms+cancel_pol+room_cat+Family_Friendly+Pets_Allowed+Smoking_Allowed+Events_Allowed+rev_rating+rev_acc+rev_clean+rev_check+rev_comm+rev_loc+rev_val+host_responsetime+host_response_rate+host_is_superhost+host_listings_count+accommodates+Basic_Amenities+deluxe_amenities+Total_Amenities+security_deposit+cleaning_fee+number_of_reviews+reviews_per_month+NearbyArson+NearbyAssault+NearbyBikeShare+NearbyClassASexOffenders+NearbyGunShots+NearbyHomicide+NearbyMetros+NearbyMuseums+NearbySexCrime+NearbySexOffenders+Closest_BikeShare+Closest_Metro+Closest_Museum+Closest_SexCrime+Closest_SexOffender+Closest_Arson+Closest_Assault+Closest_GunShot+Closest_Homicide+Closest_ClassA_SexOffender
                ,data=airbnb_data_price[traindat_p,])
summary(lm.avgprice)
lm.avgpricetest<-lm(AvgNightlyPrice2~bedrooms+bathrooms+cancel_pol+room_cat+Family_Friendly+Pets_Allowed+Smoking_Allowed+Events_Allowed+rev_rating+rev_acc+rev_clean+rev_check+rev_comm+rev_loc+rev_val+host_responsetime+host_response_rate+host_is_superhost+host_listings_count+accommodates+Basic_Amenities+deluxe_amenities+Total_Amenities+security_deposit+cleaning_fee+number_of_reviews+reviews_per_month+NearbyArson+NearbyAssault+NearbyBikeShare+NearbyClassASexOffenders+NearbyGunShots+NearbyHomicide+NearbyMetros+NearbyMuseums+NearbySexCrime+NearbySexOffenders+Closest_BikeShare+Closest_Metro+Closest_Museum+Closest_SexCrime+Closest_SexOffender+Closest_Arson+Closest_Assault+Closest_GunShot+Closest_Homicide+Closest_ClassA_SexOffender
                    ,data=airbnb_data_price[-traindat_p,])
summary(lm.avgpricetest)
stepavg <- stepAIC(lm.avgprice, direction="both")
stepavg$anova
colsavg=AvgNightlyPrice2 ~ bedrooms + bathrooms + room_cat + Family_Friendly + 
  Pets_Allowed + Smoking_Allowed + Events_Allowed + rev_rating + 
  rev_clean + rev_comm + rev_loc + rev_val + host_responsetime + 
  host_response_rate + host_is_superhost + accommodates + Basic_Amenities + 
  deluxe_amenities + cleaning_fee + number_of_reviews + reviews_per_month + 
  NearbyAssault + NearbyBikeShare + NearbyClassASexOffenders + 
  Closest_Metro + Closest_Museum + NearbyAssault + Closest_SexOffender + Closest_Arson + 
  Closest_GunShot
lm.priceavg2<-lm(colsavg,data=airbnb_data_price[traindat_p,])
summary(lm.priceavg2)
plot(lm.priceavg2)
####################################################################################
#Analysis for rating
lm.overallrating<-lm(rev_rating~rev_acc+rev_clean+rev_check+rev_comm+rev_loc+rev_val
                     ,data=airbnb_data[traindat,])
stepoverallrating <- stepAIC(lm.overallrating, direction="both")
stepoverallrating$anova
colsoverallrating=rev_rating ~ rev_acc + rev_clean + rev_check + rev_comm + rev_loc + 
  rev_val
summary(lm.overallrating)
plot(lm.overallrating)
lm.overallratingtest<-lm(colsoverallrating,data=airbnb_data[-traindat,])
summary(lm.overallratingtest)
ggplot(airbnb_data, aes(x=(rev_rating2^3))) + geom_histogram()
lm.overallrating2<-lm(z3~rev_acc+rev_clean+rev_check+rev_comm+rev_loc+rev_val
                      ,data=airbnb_data[traindat,])
rf.rating=randomForest(rev_rating~rev_acc+rev_clean+rev_check+rev_comm+rev_loc+rev_val
                       ,data=airbnb_data[traindat,],mtry=1,importance=TRUE)
yhat.rfrating = predict(rf.rating,data=airbnb_data[-traindat,])
mean((yhat.rfrating-airbnb_data[-traindat,"rev_rating"])^2)
rev_rating2=log10(airbnb_data$rev_rating)
ggplot(airbnb_data, aes(x=rev_rating2)) + geom_histogram()
####################################################################################
#Analysis for communication rating
lm.commrating<-lm(rev_comm~cancel_pol+host_responsetime+host_response_rate+host_is_superhost+host_listings_count+number_of_reviews+reviews_per_month+len_desc+len_summary+len_rules
                  ,data=airbnb_data[traindat,])
stepcommrating <- stepAIC(lm.commrating, direction="both")
stepcommrating$anova
colscommrating=rev_comm ~ cancel_pol + host_response_rate + host_is_superhost + 
  host_listings_count + reviews_per_month + len_desc + len_summary + 
  len_rules
lm.commratingtest<-lm(colscommrating,data=airbnb_data[-traindat,])
summary(lm.commratingtest)
########################################################################################
#Analysis for value rating
lm.valrating<-lm(rev_val~bedrooms+bathrooms+cancel_pol+room_cat+Family_Friendly+Pets_Allowed+Smoking_Allowed+Events_Allowed+accommodates+Basic_Amenities+deluxe_amenities+Total_Amenities+security_deposit+cleaning_fee+number_of_reviews+reviews_per_month+NearbyArson+NearbyAssault+NearbyBikeShare+NearbyClassASexOffenders+NearbyGunShots+NearbyHomicide+NearbyMetros+NearbyMuseums+NearbySexCrime+NearbySexOffenders+Closest_BikeShare+Closest_Metro+Closest_Museum+Closest_SexCrime+Closest_SexOffender+Closest_Arson+Closest_Assault+Closest_GunShot+Closest_Homicide+Closest_ClassA_SexOffender
                 +AvgNightlyPrice2+ Cost_room +Cost_person_included
                 ,data=airbnb_data[traindat,])
summary(lm.valrating)
stepvalrating <- stepAIC(lm.valrating, direction="both")
stepvalrating$anova
colsvalrating=rev_val ~ bathrooms + cancel_pol + room_cat + Pets_Allowed + 
  Smoking_Allowed + Events_Allowed + accommodates + Basic_Amenities + 
  Total_Amenities + security_deposit + cleaning_fee + reviews_per_month + 
  NearbyGunShots + Closest_BikeShare + Closest_Metro + 
  Closest_SexOffender + Closest_GunShot + Closest_Homicide + 
  Cost_room
lm.valratingtest<-lm(colsvalrating,data=airbnb_data[-traindat,])
summary(lm.valratingtest)

####################################################################################
#Analysis for cleanliness rating
lm.cleanrating<-lm(rev_clean~Family_Friendly+Pets_Allowed+Smoking_Allowed+Events_Allowed+host_listings_count+number_of_reviews+reviews_per_month+Basic_Amenities+deluxe_amenities+Total_Amenities+security_deposit+cleaning_fee
                   ,data=airbnb_data[traindat,])
summary(lm.cleanrating)
stepcleanrating <- stepAIC(lm.cleanrating, direction="both")
stepcleanrating$anova
colscleanrating=rev_clean ~ Family_Friendly + Pets_Allowed + Smoking_Allowed + 
  Events_Allowed + host_listings_count + reviews_per_month + 
  Total_Amenities + cleaning_fee
lm.cleanratingtest<-lm(colscleanrating,data=airbnb_data[-traindat,])
summary(lm.cleanratingtest)


##################
#K-means with k=2
#K-means clustering depends on the initial cluster assignment
#By setting nstart=25, k-means are performed with 25 different 
#initial assignment and report the best result.
km.out = kmeans(airbnb_data, 2, nstart = 25)

# Save cluster numbers - as.factor to differentiate clusters
km.clusters = as.factor(km.out$cluster)
#table(km.clusters,hc.clusters)

#ggplot(airbnb_data, aes(rev_rating, AvgNightlyPrice, color = km.clusters)) + geom_point()
fviz_cluster(km.out, data = airbnb_data)

k2 = kmeans(airbnb_data[,c("AvgNightlyPrice","deluxe_amenities")], centers = 2, nstart = 25)
str(k2)

# Visualize clusters
#fviz_cluster(k2, data = airbnb_data[,c(25,54)], geom="point")
fviz_cluster(k2, data = airbnb_data, geom="point", choose.vars = c("AvgNightlyPrice", "Closest_Arson"), stand = FALSE) + theme_bw()
#If there are more than two dimensions (variables) fviz_cluster will 
# perform principal component analysis and plot the data points 
# according to the first two principal components that explain the 
# majority of the variance.

k3 = kmeans(airbnb_data[,c("AvgNightlyPrice","Closest_Arson")], centers = 3, nstart = 25)
k4 = kmeans(airbnb_data[,c("AvgNightlyPrice","Closest_Arson")], centers = 4, nstart = 25)
k5 = kmeans(airbnb_data[,c("AvgNightlyPrice","Closest_Arson")], centers = 5, nstart = 25)

# plots to compare
p1 = fviz_cluster(k2, data = airbnb_data[,c("AvgNightlyPrice","Closest_Arson")],geom = "point") + ggtitle("k = 2")
p2 = fviz_cluster(k3, data = airbnb_data[,c("AvgNightlyPrice","Closest_Arson")],geom = "point") + ggtitle("k = 3")
p3 = fviz_cluster(k4, data = airbnb_data[,c("AvgNightlyPrice","Closest_Arson")],geom = "point") + ggtitle("k = 4")
p4 = fviz_cluster(k5, data = airbnb_data[,c("AvgNightlyPrice","Closest_Arson")],geom = "point") + ggtitle("k = 5")

grid.arrange(p1, p2, p3, p4, nrow = 2)

# Compute total within-cluster sum of square 
wss = function(k) {
  kmeans(airbnb_data, k, nstart = 10 )$tot.withinss
}

# Compute and plot wss for k = 2 to k = 10
k.values = 2:10

# extract wss for 2-10 clusters
wss_values = map_dbl(k.values, wss)

plot(k.values, wss_values,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares",
     main="Sum of Squares vs Cluster Number")

#One way to determine k is to choose the k where you see a bend.

