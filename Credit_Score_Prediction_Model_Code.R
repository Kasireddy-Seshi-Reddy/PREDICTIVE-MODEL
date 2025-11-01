library(caret)
library(e1071)
library(randomForest)
library(class)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(dplyr)

data <- read.csv("credit_score_data.csv")
str(data)
summary(data)

colSums(is.na(data))
data$Age[is.na(data$Age)] <- mean(data$Age, na.rm = TRUE)
data$Annual_Income[is.na(data$Annual_Income)] <- median(data$Annual_Income, na.rm = TRUE)
data$Credit_Score <- as.factor(data$Credit_Score)
data$Gender <- as.factor(data$Gender)
data$Occupation <- as.factor(data$Occupation)
numeric_cols <- sapply(data, is.numeric)
data[numeric_cols] <- scale(data[numeric_cols])

set.seed(123)
trainIndex <- createDataPartition(data$Credit_Score, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

ctrl <- trainControl(method = "cv", number = 5)
model_knn <- train(Credit_Score ~ ., data = trainData, method = "knn", trControl = ctrl)
pred_knn <- predict(model_knn, testData)
acc_knn <- confusionMatrix(pred_knn, testData$Credit_Score)$overall['Accuracy']

model_nb <- naiveBayes(Credit_Score ~ ., data = trainData)
pred_nb <- predict(model_nb, testData)
acc_nb <- confusionMatrix(pred_nb, testData$Credit_Score)$overall['Accuracy']

model_dt <- rpart(Credit_Score ~ ., data = trainData, method = "class")
pred_dt <- predict(model_dt, testData, type = "class")
acc_dt <- confusionMatrix(pred_dt, testData$Credit_Score)$overall['Accuracy']
rpart.plot(model_dt)

model_rf <- randomForest(Credit_Score ~ ., data = trainData, ntree = 100, importance = TRUE)
pred_rf <- predict(model_rf, testData)
acc_rf <- confusionMatrix(pred_rf, testData$Credit_Score)$overall['Accuracy']

accuracy_results <- data.frame(
  Model = c("KNN", "Naive Bayes", "Decision Tree", "Random Forest"),
  Accuracy = c(acc_knn, acc_nb, acc_dt, acc_rf)
)
print(accuracy_results)

ggplot(accuracy_results, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", color = "black") +
  theme_minimal() +
  ggtitle("Model Accuracy Comparison") +
  geom_text(aes(label = round(Accuracy, 3)), vjust = -0.5)

importance(model_rf)
varImpPlot(model_rf)

best_model <- accuracy_results[which.max(accuracy_results$Accuracy), ]
cat("Best Performing Model:", best_model$Model, "with Accuracy:", round(best_model$Accuracy, 3))
