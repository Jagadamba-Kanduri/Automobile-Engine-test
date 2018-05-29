rm(list = ls(all=TRUE))
#setwd("C:/Users/lenovo/Desktop/Insofee/SelfPractice/automobile_engine")

#loading the file ----

#loading train data
raw<- read.csv("train.csv",header = TRUE)
raw_pass <- read.csv("Train_AdditionalData.csv",header = TRUE)

TestA=c();TestB=c();

for (i in raw$ID) {
  TestA=c(TestA,i %in% raw_pass$TestA)
  TestB=c(TestB,i %in% raw_pass$TestB)
}
raw_pass <- data.frame(cbind(raw$ID,TestA,TestB))
colnames(raw_pass) <- c("ID","TestA","TestB")

train <- merge(x=raw,y=raw_pass,by="ID",all.x=TRUE) 
#Reading test data
test <- read.csv("test.csv",header = TRUE)
test_pass <- read.csv("Test_AdditionalData.csv",header = TRUE)

TestA=c();TestB=c();
for (i in test$ID) {
  TestA=c(TestA,i %in% test_pass$TestA)
  TestB=c(TestB,i %in% test_pass$TestB)
}
test_pass <- data.frame(cbind(test$ID,TestA,TestB))
colnames(test_pass) <- c("ID","TestA","TestB")

test <- merge(x=test,y=test_pass,by="ID",all.x=TRUE) 

#Removing the un nessisary variables
rm(list = setdiff(ls(), c("train","test")))
#Exploring data ----
summary(train)
str(train)

train$Number.of.Cylinders <- as.factor(train$Number.of.Cylinders)
train$TestA <- as.factor(train$TestA)
train$TestB <- as.factor(train$TestB)

test$Number.of.Cylinders <- as.factor(test$Number.of.Cylinders)
test$TestA <- as.factor(test$TestA)
test$TestB <- as.factor(test$TestB)
#removing ID
train$ID <- NULL
test$ID <- NULL
sum(is.na(train))/nrow(train)
colSums(is.na(train))
sum(!complete.cases(train))

prop.table(table(train$y)) # Checking class imbalence

#spliting the train data in to validation and train using caret
library(caret)
set.seed(123)

train_rows <- createDataPartition(train$y , p = 0.7,list = F)
train <- train[train_rows, ]
valid <- train[-train_rows, ]

#Imputing the data using central imputation
library(DMwR)
train <- centralImputation(train)
valid <- centralImputation(valid)
test <- centralImputation(test)

#Learning curves ----

nsteps = 30
learnCurve <- data.frame(datasamples = integer(nsteps),
                         trainMetric = integer(nsteps),
                         testMetric = integer(nsteps))
# test data response feature
testY <- valid$y

# Run algorithms using 5-fold cross validation with 1 repeats
trainControl <- trainControl(method="repeatedcv", number=3, repeats=1)
metric <- "Accuracy"

nrows <- nrow(train)
# loop over training examples
for (j in 1:nsteps*floor(nrows/nsteps)) {
  i = j/floor(nrows/nsteps)
  cat(i)
  learnCurve$datasamples[i] <- j
  
  # train learning algorithm with size i
  sampledData <- train[sample(j),]
  fit.svm <- train(y~., 
                   data=sampledData, 
                   method="treebag", 
                   metric=metric,
                  # preProc=c("center", "scale"), 
                  trControl=trainControl)        
  
  prediction <- predict(fit.svm, newdata = sampledData)
  
  Metric <- postResample(prediction, sampledData$y)
  
  learnCurve$trainMetric[i] <- Metric[1]
  # use trained parameters to predict on test data
  prediction <- predict(fit.svm, newdata = valid)
  Metric <- postResample(prediction, testY)
  learnCurve$testMetric[i] <- Metric[1]
}
library(ggplot2)
p <- ggplot(learnCurve)
p <- p + aes(x = datasamples) + 
  geom_line(aes(y = trainMetric, colour='train Metric')) + 
  geom_line(aes(y = testMetric, colour='test Metric'))
set.seed(123)
ggsave("myPlot1.jpg", plot=p)



#Logistic regression ----

logitMod <- glm(y ~ ., data=train, family=binomial(link="logit"))
predicted <- predict(logitMod, train,type = "response")

#Apply ROC for better cutoff ----
library(ROCR)
library(ggplot2)
prob <- prediction(predicted,train$y)
eval <- performance(prob, "acc")

#identify best cutoff
max <- which.max(slot(eval,"y.values")[[1]])
acc <- slot(eval,"y.values")[[1]][max]
cut <- slot(eval,"x.values")[[1]][max]
print(c(Accuracy=acc,CutOff=cut))

# Getting the true positive rate and false negative rate
tprfpr <- performance(prob, "tpr", "fpr")
plot(tprfpr,
     colorize=T,
     main = "ROC Curve",
     ylab="Sensitivity",
     xlab="1-Specifiity")
abline(a=0,b=1)

## Area under the curve
aucvalue <- performance(prob, measure = "auc");
aucvalue@y.values[[1]]  

legend(.8,.4,round(aucvalue@y.values[[1]],2),title = "AUC")

# predicting logistic regression model where cutoff >0.5----
train_glm <- ifelse(predicted > 0.5, "pass","fail")
confusionMatrix(train$y,train_glm)

predicted <- predict(logitMod, valid,type = "response")
valid_glm <- ifelse(predicted > 0.5, "pass","fail")
confusionMatrix(valid$y,valid_glm)

predicted <- predict(logitMod, test,type = "response")
test_glm <- ifelse(predicted > 0.5, "pass","fail")

#Navie Bayers ----
library(e1071)
NB_Model=naiveBayes(y ~., data=train)

train_NB=predict(NB_Model,train)
confusionMatrix(train$y,train_NB)

valid_NB <- predict(NB_Model,valid)
confusionMatrix(valid$y,valid_NB)

test_NB=predict(NB_Model,test)

# Decision Trees----

library(rpart)
dtCart0=rpart(y~.,
             data=train,
             method="class",
             parms = list(split = 'information'),
             control = rpart.control(cp=.0002,minsplit = 5,minbucket = 5,maxdepth = 10,xval = 10))
               


#Optimising cp value
printcp(dtCart) # Consiser cp = 0.00654206 / 
#1 sd value 0.00218069(0.31308+0.015756), take the cp values xerror + xstd ,
#look for max error which is less than xerror + xstd
plotcp(dtCart)

cp = dtCart$cptable[which.min(dtCart$cptable[,"xerror"])]
dtCart = prune(dtCart0,cp=0.006542056) # Prune tree

library(rpart.plot)
rpart.plot(dtCart,fallen.leaves = T,cex = 0.7,extra = 1) 
summary(dtCart)
DT_train <- predict(dtCart,train,type = "class")
confusionMatrix(DT_train,train$y)$overall[1]
DT_test <- predict(dtCart,test,type = "class")
                         
## Need to check the below
#library(C50)
#dtC50= C5.0(y ~ ., data = train, rules=TRUE)
#summary(dtC50)
#C5imp(dtC50, pct=TRUE)
#C50_train <- predict(dtC50,train,type = "class")
#confusionMatrix(train$y,C50_train)

#Random Forest ----
library(randomForest)
model_rf0 <- randomForest(y ~ ., data=train, keep.forest=TRUE)
print(model_rf0) 
model_rf0$predicted 
model_rf0$importance

varImpPlot(model_rf0)

#Error rate
plot(model_rf0) 

#Tuning mtry
t <- tuneRF(train[,-1],train[,1],
       stepFactor = 0.5,
       plot = TRUE,
       ntreeTry = 200,
       trace = TRUE,
       improve = 0.05)

model_rf <- randomForest(y ~ ., 
                         data=train, 
                         ntree = 200,
                         mtry = 4 ,
                         importance=TRUE,
                         proximity = TRUE)
print(model_rf)
#No of nodes for the tree
hist(treesize(model_rf))

varImpPlot(model_rf,sort = T,n.var = 10,
           main = "Top 10 Importent variables") # Variable importence

importance(model_rf)
varUsed(model_rf)

MDSplot(model_rf,train$y)

RF_train <- predict(model_rf,train,type = "class")
confusionMatrix(RF_train,train$y)$overall[1]
RF_test <- predict(model_rf,test)

### Grid search----

control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
set.seed(1234)
tunegrid <- expand.grid(.mtry=c(1:15))
rf_gridsearch <- train(y~., data=train, 
                       method="rf", metric=metric, 
                       tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)
##--##


### SVM ----

SVM_model <- svm(y~., data=train,kernel = "radial")
summary(SVM_model)


# Tuning by grid search
set.seed(123)
tmodel <- tune(svm, y~., data=train,
               ranges = list(gamma=10^(-6:-1),cost =2^(2:3)))
plot(tmodel)
summary(tmodel)
# best Model
mymodel <- tmodel$best.model
summary(mymodel)

train_svm <- predict(mymodel, train)
tab <- table(train_svm,train$y)

missclassErr <- 1-confusionMatrix(train_svm,train$y)$overall[1]
missclassErr

  #Stacking----
# Generate level-one dataset for training the ensemble metalearner
predRF <- predict(modelFitRF, newdata = validation)
predGBM <- predict(modelFitGBM, newdata = validation)
prefLDA <- predict(modelFitLDA, newdata = validation)
predDF <- data.frame(predRF, predGBM, prefLDA, diagnosis = validation$diagnosis, stringsAsFactors = F)

# Train the ensemble
modelStack <- train(diagnosis ~ ., data = predDF, method = "glm")
# Generate predictions on the test set
testPredRF <- predict(modelFitRF, newdata = testing)
testPredGBM <- predict(modelFitGBM, newdata = testing)
testPredLDA <- predict(modelFitLDA, newdata = testing)

# Using the base learner test set predictions, 
# create the level-one dataset to feed to the ensemble
testPredLevelOne <- data.frame(testPredRF, testPredGBM, testPredLDA, diagnosis = testing$diagnosis, stringsAsFactors = F)
combPred <- predict(modelStack, testPredLevelOne)

# Evaluate ensemble test performance
confusionMatrix(combPred, testing$diagnosis)$overall[1]

# Evaluate base learner test performance 
confusionMatrix(testPredRF, testing$diagnosis)$overall[1]
confusionMatrix(testPredGBM, testing$diagnosis)$overall[1]
confusionMatrix(testPredLDA, testing$diagnosis)$overall[1]
