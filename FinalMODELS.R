# Load required libraries
library(randomForest)
library(rpart)
library(e1071)
library(xgboost)
library(gbm)
library(caret)
library(glmnet)
library(infotheo)  # For mutual information calculation
library(reshape2)  # For data manipulation
library(ggplot2)   # For plotting

# Read the data
data <- read.csv('farm_rice.csv')

# Remove unimportant features
unimportant_features <- c('id', 'bimas', 'working_labor', 'working_family_labor')
data <- data[, !names(data) %in% unimportant_features]

# Compute mutual information between features and target variable
mi <- apply(data[, -which(names(data) == "Gross_output_kg")], 2, mutual_info_regression, data$Gross_output_kg)

# Select features based on a threshold (e.g., top 5 features)
selected_features <- names(data)[order(-mi)][1:5]

# Subset the data with selected features
selected_data <- data[, c(selected_features, "Gross_output_kg")]

# Print selected features
print("Selected features based on mutual information:")
print(selected_features)

# Split selected data into train and test sets
set.seed(123)
train_indices <- createDataPartition(selected_data$Gross_output_kg, p = 0.8, list = FALSE) 
train_data <- selected_data[train_indices, ]
test_data <- selected_data[-train_indices, ]

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

# Print Random Forest metrics
cat("Random Forest Metrics:\n")
cat("RMSE:", rf_metrics$RMSE, "\n")
cat("MSE:", rf_metrics$MSE, "\n")
cat("R-squared:", rf_metrics$R_squared, "\n\n")

# Train decision tree model
dt_model <- rpart(Gross_output_kg ~ ., data = train_data)

# Make predictions with Decision Tree
dt_predictions_test <- predict(dt_model, newdata = test_data)

# Calculate metrics for Decision Tree
dt_metrics <- calculate_metrics(test_data$Gross_output_kg, dt_predictions_test)

# Print Decision Tree metrics
cat("Decision Tree Metrics:\n")
cat("RMSE:", dt_metrics$RMSE, "\n")
cat("MSE:", dt_metrics$MSE, "\n")
cat("R-squared:", dt_metrics$R_squared, "\n\n")

# Train SVM model
svm_model <- svm(Gross_output_kg ~ ., data = train_data)

# Make predictions with SVM
svm_predictions_test <- predict(svm_model, newdata = test_data)

# Calculate metrics for SVM
svm_metrics <- calculate_metrics(test_data$Gross_output_kg, svm_predictions_test)

# Print SVM metrics
cat("SVM Metrics:\n")
cat("RMSE:", svm_metrics$RMSE, "\n")
cat("MSE:", svm_metrics$MSE, "\n")
cat("R-squared:", svm_metrics$R_squared, "\n\n")

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

# Print XGBoost metrics
cat("XGBoost Metrics:\n")
cat("RMSE:", xgb_metrics$RMSE, "\n")
cat("MSE:", xgb_metrics$MSE, "\n")
cat("R-squared:", xgb_metrics$R_squared, "\n\n")

# Train Gradient Boosting Machine (GBM) model
gbm_model <- gbm(Gross_output_kg ~ ., data = train_data, distribution = "gaussian", n.trees = 500, interaction.depth = 6)

# Make predictions with GBM
gbm_predictions_test <- predict(gbm_model, newdata = test_data, n.trees = 500)

# Calculate metrics for GBM
gbm_metrics <- calculate_metrics(test_data$Gross_output_kg, gbm_predictions_test)

# Print GBM metrics
cat("GBM Metrics:\n")
cat("RMSE:", gbm_metrics$RMSE, "\n")
cat("MSE:", gbm_metrics$MSE, "\n")
cat("R-squared:", gbm_metrics$R_squared, "\n\n")

# Train Lasso Regression model
lasso_model <- cv.glmnet(x = as.matrix(train_data[, -which(names(train_data) == "Gross_output_kg")]), 
                         y = train_data$Gross_output_kg, 
                         alpha = 1, 
                         nfolds = 10)

# Make predictions with Lasso Regression
lasso_predictions_test <- predict(lasso_model, newx = as.matrix(test_data[, -which(names(test_data) == "Gross_output_kg")]))

# Calculate metrics for Lasso Regression
lasso_metrics <- calculate_metrics(test_data$Gross_output_kg, lasso_predictions_test)

# Print Lasso Regression metrics
cat("Lasso Regression Metrics:\n")
cat("RMSE:", lasso_metrics$RMSE, "\n")
cat("MSE:", lasso_metrics$MSE, "\n")
cat("R-squared:", lasso_metrics$R_squared, "\n\n")

# Train Ridge Regression model
ridge_model <- cv.glmnet(x = as.matrix(train_data[, -which(names(train_data) == "Gross_output_kg")]), 
                         y = train_data$Gross_output_kg, 
                         alpha = 0, 
                         nfolds = 10)

# Make predictions with Ridge Regression
ridge_predictions_test <- predict(ridge_model, newx = as.matrix(test_data[, -which(names(test_data) == "Gross_output_kg")]))

# Calculate metrics for Ridge Regression
ridge_metrics <- calculate_metrics(test_data$Gross_output_kg, ridge_predictions_test)

# Print Ridge Regression metrics
cat("Ridge Regression Metrics:\n")
cat("RMSE:", ridge_metrics$RMSE, "\n")
cat("MSE:", ridge_metrics$MSE, "\n")
cat("R-squared:", ridge_metrics$R_squared, "\n\n")

# Combine metrics into a data frame
metrics <- data.frame(
  Model = c("Random Forest", "Decision Tree", "SVM", "XGBoost", "GBM", "Lasso Regression", "Ridge Regression"),
  RMSE = c(rf_metrics$RMSE, dt_metrics$RMSE, svm_metrics$RMSE, xgb_metrics$RMSE, gbm_metrics$RMSE, lasso_metrics$RMSE, ridge_metrics$RMSE),
  MSE = c(rf_metrics$MSE, dt_metrics$MSE, svm_metrics$MSE, xgb_metrics$MSE, gbm_metrics$MSE, lasso_metrics$MSE, ridge_metrics$MSE),
  R_squared = c(rf_metrics$R_squared, dt_metrics$R_squared, svm_metrics$R_squared, xgb_metrics$R_squared, gbm_metrics$R_squared, lasso_metrics$R_squared, ridge_metrics$R_squared)
)

# Melt the data frame for plotting
metrics_melted <- melt(metrics, id.vars = "Model")

# Plotting
ggplot(metrics_melted, aes(x = Model, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~variable, scales = "free_y", ncol = 1) +
  labs(title = "Comparison of Model Performance Metrics",
       y = "Value",
       fill = "Metric") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(legend.title = element_text(size = 12, face = "bold")) +
  theme(legend.text = element_text(size = 10))
  
