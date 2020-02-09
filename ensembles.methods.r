#####   MADE BY    ######
# 07/01/20
#Created by: Paraskevi Sifnaiou 
#####################

############# Step 1: Data import ##############
setwd("C:/Users/sifne/Desktop/")

# Load libraries
library(readxl)
library(tidyverse)
library(caretEnsemble)
library(magrittr)
library(plyr)
library(caret)
library(ROCR)

online_shoppers_intention <- read.csv('online_shoppers_intention.csv')
boxplot(online_shoppers_intention)
onl_shop<-online_shoppers_intention #to preserve the original dataset

#Remove columns:11-17 to get higher accuracy
onl_shop[11:17] <- NULL
boxplot(onl_shop)

#check of NA's
sum(is.na(onl_shop))
summary(onl_shop)
str(onl_shop)

#Preprocess
onl_shop$Revenue <- make.names(onl_shop$Revenue)
onl_shop$Revenue<-as.factor(onl_shop$Revenue)
glimpse(onl_shop)

#############    Data Split    ###############

#Create index to split   
index <- createDataPartition(onl_shop$Revenue,p=0.75,list=FALSE)
# Subset training set with index
onl_shop.training<-onl_shop[index,]
# Subset test set with index
onl_shop.test<-onl_shop[-index,]


##repeated k-fold cross validation
# define training control
control <- trainControl(method="repeatedcv", number=10, repeats=3,
                        savePredictions=TRUE, classProbs=TRUE,preProc = c("center","scale"))

####################### bagging algorithm #############################

seed <- 7
metric <- "Accuracy"

# Treebag 
set.seed(seed)
fit.treebag <- train(Revenue~., data=onl_shop.training, method="treebag", metric=metric, trControl=control)

# Random Forest
set.seed(seed)
fit.rf <- train(Revenue~., data=onl_shop.training, method="rf", metric=metric, trControl=control)

#treebag testing set accuracy
predictions_treebag<-predict(object=fit.treebag ,onl_shop.test, type="raw")
table(predictions_treebag)
confusionMatrix(predictions_treebag,onl_shop.test$Revenue)

#random forest testing set accuracy
predictions_rf<-predict(object=fit.rf ,onl_shop.test, type="raw")
table_rf<-table(predictions_rf)
table_rf
confusionMatrix(predictions_rf,onl_shop.test$Revenue)

# summarize results
bagging_results <- resamples(list(treebag=fit.treebag, rf=fit.rf))
summary(bagging_results)
dotplot(bagging_results)

######################### stacking algorithm ###################################

# create submodels
algorithmList <- c( 'rpart', 'knn' ,'nb')
set.seed(seed)
models <- caretList(Revenue~., data=onl_shop.training, trControl=control, methodList=algorithmList)

#results
results <- resamples(models)
summary(results)
dotplot(results)

# correlation between results
modelCor(results)
splom(results)

################## Stack 2 models with best correlations: only knn & nb  ############


#knn
set.seed(seed)
stack.knn <- caretStack(models, method="knn", metric="Accuracy", trControl=control)
print(stack.knn)

prediction_knn2<-predict(stack.knn ,onl_shop.test)
table(prediction_knn2)
confusionMatrix(prediction_knn2,onl_shop.test$Revenue)

#nb
set.seed(seed)
stack.nb <- caretStack(models, method="nb", metric="Accuracy", trControl=control)
print(stack.nb)

prediction_nb2<-predict(stack.nb ,onl_shop.test)
table(prediction_nb2)
confusionMatrix(prediction_nb2,onl_shop.test$Revenue)


###################### Measure Performance ##############################


####random forest#####
#AUC
pred_rf_raw <- predict(fit.rf,onl_shop.test,type = "raw")
prediction_raw <-  prediction(as.numeric(predictions_rf), onl_shop.test$Revenue)
tpr_fpr <- performance(prediction_raw,"tpr","fpr")
plot(tpr_fpr)

plot(tpr_fpr,colorize=TRUE,main = "ROC Curve",
     ylab = "sensivity",
     xlab = "specifity") 
abline(a= 0,b=1)

pred_auc  <- performance(prediction_raw ,measure="auc")
pred_auc 

auc_value <- pred_auc@y.values[[1]]
auc_value <- round(auc_value,4)
legend(.6,.4,auc_value,title = "AUC")

######treebag########
#AUC
pred_rf_raw <- predict(fit.treebag,onl_shop.test,type = "raw")
prediction_raw <-  prediction(as.numeric(predictions_treebag), onl_shop.test$Revenue)
tpr_fpr <- performance(prediction_raw,"tpr","fpr")

plot(tpr_fpr,colorize=TRUE,main = "ROC Curve",
     ylab = "sensivity",
     xlab = "specifity") 
abline(a= 0,b=1)

pred_auc  <- performance(prediction_raw ,measure="auc")
pred_auc 

auc_value <- pred_auc@y.values[[1]]
auc_value <- round(auc_value,4)
legend(.6,.4,auc_value,title = "AUC")

########knn############
set.seed(seed)
pred_knn_prob<- predict(stack.knn,onl_shop.test,type = "prob")
prediction_knn <- prediction( pred_knn_prob, onl_shop.test$Revenue)

#evalutation of accuracy
accuracy <- performance(prediction_knn,"acc")
plot(accuracy)

#precision vs recall
prec_vs_recall <- performance(prediction_knn,"prec","rec")
plot(prec_vs_recall)

#AUC
pred_knn_raw <- predict(stack.knn,onl_shop.test,type = "raw")
prediction_raw <-  prediction(as.numeric(pred_knn_raw), onl_shop.test$Revenue)
tpr_fpr <- performance(prediction_raw,"tpr","fpr")
plot(tpr_fpr)

plot(tpr_fpr,colorize=TRUE,main = "ROC Curve",
     ylab = "sensivity",
     xlab = "specifity") 
abline(a= 0,b=1)

pred_auc  <- performance(prediction_raw ,measure="auc")
pred_auc 

auc_value <- pred_auc@y.values[[1]]
auc_value <- round(auc_value,4)
legend(.6,.4,auc_value,title = "AUC")

#########nb###########
set.seed(seed)
pred_nb_prob<- predict(stack.nb,onl_shop.test,type = "prob")
prediction_nb <- prediction( pred_nb_prob, onl_shop.test$Revenue)

#evalutation of accuracy
accuracy <- performance(prediction_nb,"acc")
plot(accuracy)

#precision vs recall
prec_vs_recall <- performance(prediction_nb,"prec","rec")
plot(prec_vs_recall)

#AUC
pred_nb_raw <- predict(stack.nb,onl_shop.test,type = "raw")
prediction_raw <-  prediction(as.numeric(pred_nb_raw), onl_shop.test$Revenue)
tpr_fpr <- performance(prediction_raw,"tpr","fpr")
plot(tpr_fpr)

plot(tpr_fpr,colorize=TRUE,main = "ROC Curve",
     ylab = "sensivity",
     xlab = "specifity") 
abline(a= 0,b=1)

pred_auc  <- performance(prediction_raw ,measure="auc")
pred_auc 

auc_value <- pred_auc@y.values[[1]]
auc_value <- round(auc_value,4)
legend(.6,.4,auc_value,title = "AUC")

#############training and testing time#################
#bagging 
  ##treebag: training time
system.time(fit.treebag <- train(Revenue~., data=onl_shop.training, method="treebag", metric=metric, trControl=control))
  ##treebag: testing time
system.time(predictions_treebag<-predict(object=fit.treebag ,onl_shop.test, type="raw"))

  ##random forest: training time
system.time(fit.rf <- train(Revenue~., data=onl_shop.training, method="rf", metric=metric, trControl=control))
  ##random forest: testing time
system.time(predictions_rf<-predict(object=fit.rf ,onl_shop.test, type="raw"))


#stacking
    #training knn
system.time(stack.knn <- caretStack(models, method="knn", metric="Accuracy", trControl=control))
            
    #testing knn
system.time(prediction_knn2<-predict(stack.knn ,onl_shop.test))

     #training nb
system.time(stack.nb <- caretStack(models, method="nb", metric="Accuracy", trControl=control))
     
     #testing nb
system.time(pred_nb_prob<- predict(stack.nb,onl_shop.test,type = "prob"))


