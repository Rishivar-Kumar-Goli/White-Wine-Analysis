library(dplyr) # Wrangling Data
library(caret) # Confussion Matrix
library(FactoMineR) #
library(e1071)# Naive Bayes
library(ROCR) # ROC
library(partykit) # Decision Tree
library(rsample)
library(magrittr)

white_Wine <- read.csv("winequality-white.csv" , sep = ";")

summary(white_Wine)

unique(white_Wine$quality)


anyNA(white_Wine)

white_Wine <- white_Wine %>% 
  mutate(quality=as.factor(ifelse(quality>6,"Excellent","Poor-Normal")))
white_Wine

prop.table(table(white_Wine$quality))

set.seed(211)
wine_dsamp <- downSample(x= white_Wine %>% select(-quality),
                         y=white_Wine$quality,
                         yname = "quality")
prop.table(table(wine_dsamp$quality))

set.seed(417)

split <- initial_split(data = white_Wine, prop = 0.8, strata = "quality")

train <- training(split)
test <- testing(split)
prop.table(table(train$quality))

# model building
naive <- naiveBayes(wine_dsamp %>% select(-quality), wine_dsamp$quality, laplace = 1)
# model fitting
naive_pred <- predict(naive, test, type = "class") # for the class prediction

# result
confusionMatrix(naive_pred,test$quality,positive = "Excellent")

naive_prob <- predict(naive, newdata = test,type = "raw")

wine_roc <- prediction(predictions = naive_prob[,1],# prob kelas positif
                       labels = as.numeric(test$quality =="Excellent"))


perf <- performance(prediction.obj = wine_roc,
                    measure = "tpr",
                    x.measure = "fpr")

plot(perf)
abline(0,1, lty = 2)

auc <- performance(prediction.obj = wine_roc, 
                   measure = "auc")
auc@y.values

model_dt <- ctree(quality~.,white_Wine)

dtree_pred <- predict(model_dt, test, type = "response")
confusionMatrix(dtree_pred,reference = test$quality, positive = "Excellent")

pred_dt_train <- predict(model_dt, newdata = wine_dsamp, type = "response")
confusionMatrix(pred_dt_train, wine_dsamp$quality, positive = "Excellent")

model_dt_recap <- c("test", "wine_train_dsample")
Accuracy <- c(0.8746,0.8046)
Recall <- c(0.4186,0.4931)

tabelmodelrecap <- data.frame(model_dt_recap,Accuracy,Recall)

print(tabelmodelrecap)

model_dt_tun <- ctree(quality ~ ., wine_dsamp,
                      control = ctree_control(mincriterion = 0.5,
                                              minsplit = 35, #40
                                              minbucket = 20)) #12

pred_dt_test_tun <- predict(model_dt_tun, newdata = test, type = "response")
confusionMatrix(pred_dt_test_tun, test$quality, positive = "Excellent")
pred_dt_train_tun <- predict(model_dt_tun, newdata = wine_dsamp, type = "response")
confusionMatrix(pred_dt_train_tun, wine_dsamp$quality, positive = "Excellent")
model_dt_recap_prun <- c("wine.test", "wine_train_down")
Accuracy_prun <- c(0.8119,0.8046)
Recall_prun <- c(0.9535,0.9673)

tabelmodelrecap2 <- data.frame(model_dt_recap_prun,Accuracy_prun,Recall_prun)

print(tabelmodelrecap2)
plot(model_dt_tun,type = "simple")

Model_Name <- c("Naive Bayes", "Decision Tree")
Accuracy <- c(0.6865,0.7398)
Sensitivity <- c(0.8140,0.7907)
Specificity <- c(0.6667,0.7319)
Precision <- c(0.2756,0.3148)

modelrecapall <- data.frame(Model_Name,Accuracy,Recall,Specificity,Precision)

print(modelrecapall)