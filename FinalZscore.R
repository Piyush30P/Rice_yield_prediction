# Load required libraries
library(randomForest)
library(rpart)
library(e1071)
library(xgboost)
library(gbm)
library(caret)
library(glmnet)

# Read the data
data <- read.csv('farm_rice.csv')

# Remove unimportant features
unimportant_features <- c('id', 'bimas', 'working_labor', 'working_family_labor')
data <- data[, !names(data) %in% unimportant_features]

# Convert categorical variables to factors
data$region <- as.integer(as.factor(data$region))
data$status_land <- as.integer(as.factor(data$status_land))
data$varieties <- as.integer(as.factor(data$varieties))

# Normalization function - Z-score normalization
z_score_normalize <- function(x) {
  return((x - mean(x)) / sd(x))
}

# Apply Z-score normalization to all features except the target variable
data[, -which(names(data) == "Gross_output_kg")] <- apply(data[, -which(names(data) == "Gross_output_kg")], 2, z_score_normalize)

# Split data into train and test sets
set.seed(123)
train_indices <- createDataPartition(data$Gross_output_kg, p = 0.8, list = FALSE) 
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Function to calculate RMSE, MSE, and R-squared
calculate_metrics <- function(actual, predicted) {
  rmse <- sqrt(mean((actual - predicted)^2))
  mse <- mean((actual - predicted)^2)
  r_squared <- 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
  return(list(RMSE = rmse, MSE = mse, R_squared = r_squared))
}

# Train Random Forest model
rf_model <- randomForest(x = train_data[, -which(names(train_data) == "Gross_output_kg")], 
                         y = train_data$Gross_output_kg, 
                         ntree = 500) 

# Make predictions with Random Forest
rf_predictions_test <- predict(rf_model, newdata = test_data[, -which(names(test_data) == "Gross_output_kg")])

# Calculate metrics for Random Forest
rf_metrics <- calculate_metrics(test_data$Gross_output_kg, rf_predictions_test)

# Train decision tree model
dt_model <- rpart(Gross_output_kg ~ ., data = train_data)

# Make predictions with Decision Tree
dt_predictions_test <- predict(dt_model, newdata = test_data)

# Calculate metrics for Decision Tree
dt_metrics <- calculate_metrics(test_data$Gross_output_kg, dt_predictions_test)

# Train SVM model
svm_model <- svm(Gross_output_kg ~ ., data = train_data)

# Make predictions with SVM
svm_predictions_test <- predict(svm_model, newdata = test_data)

# Calculate metrics for SVM
svm_metrics <- calculate_metrics(test_data$Gross_output_kg, svm_predictions_test)

# Convert data to DMatrix for XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(train_data[, -which(names(train_data) == "Gross_output_kg")]), label = train_data$Gross_output_kg)
dtest <- xgb.DMatrix(data = as.matrix(test_data[, -which(names(test_data) == "Gross_output_kg")]), label = test_data$Gross_output_kg)

# Set parameters for XGBoost
params <- list(booster = "gbtree", objective = "reg:squarederror", eta = 0.3, max_depth = 6, min_child_weight = 1, subsample = 1, colsample_bytree = 1)

# Train XGBoost model
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 10)

# Make predictions with XGBoost
xgb_predictions_test <- predict(xgb_model, dtest)

# Calculate metrics for XGBoost
xgb_metrics <- calculate_metrics(test_data$Gross_output_kg, xgb_predictions_test)

# Train Gradient Boosting Machine (GBM) model
gbm_model <- gbm(Gross_output_kg ~ ., data = train_data, distribution = "gaussian", n.trees = 500, interaction.depth = 6)

# Make predictions with GBM
gbm_predictions_test <- predict(gbm_model, newdata = test_data, n.trees = 500)

# Calculate metrics for GBM
gbm_metrics <- calculate_metrics(test_data$Gross_output_kg, gbm_predictions_test)

# Train Lasso Regression model
lasso_model <- cv.glmnet(x = as.matrix(train_data[, -which(names(train_data) == "Gross_output_kg")]), 
                         y = train_data$Gross_output_kg, 
                         alpha = 1, 
                         nfolds = 10)

# Make predictions with Lasso Regression
lasso_predictions_test <- predict(lasso_model, newx = as.matrix(test_data[, -which(names(test_data) == "Gross_output_kg")]))

# Calculate metrics for Lasso Regression
lasso_metrics <- calculate_metrics(test_data$Gross_output_kg, lasso_predictions_test)

# Train Ridge Regression model
ridge_model <- cv.glmnet(x = as.matrix(train_data[, -which(names(train_data) == "Gross_output_kg")]), 
                         y = train_data$Gross_output_kg, 
                         alpha = 0, 
                         nfolds = 10)

# Make predictions with Ridge Regression
ridge_predictions_test <- predict(ridge_model, newx = as.matrix(test_data[, -which(names(test_data) == "Gross_output_kg")]))

# Calculate metrics for Ridge Regression
ridge_metrics <- calculate_metrics(test_data$Gross_output_kg, ridge_predictions_test)

# Create a data frame for metrics
models <- c("Random Forest", "Decision Tree", "SVM", "XGBoost", "GBM", "Lasso Regression", "Ridge Regression")
metrics_df <- data.frame(models, t(sapply(list(rf_metrics, dt_metrics, svm_metrics, xgb_metrics, gbm_metrics, lasso_metrics, ridge_metrics), unlist)))

# Print the normalized metrics table
print(metrics_df)
