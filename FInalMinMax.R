# Load required libraries
library(randomForest)
library(rpart)
library(e1071)
library(xgboost)
library(gbm)
library(caret)
library(glmnet)
library(ggplot2)
library(reshape2)

# Read the data
data <- read.csv('farm_rice.csv')

# Remove unimportant features
unimportant_features <- c('id', 'bimas', 'working_labor', 'working_family_labor')
data <- data[, !names(data) %in% unimportant_features]

# Convert categorical variables to factors
data$region <- as.integer(as.factor(data$region))
data$status_land <- as.integer(as.factor(data$status_land))
data$varieties <- as.integer(as.factor(data$varieties))

# Normalization function - Min-Max normalization
min_max_normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Apply Min-Max normalization to all features except the target variable
data[, -which(names(data) == "Gross_output_kg")] <- apply(data[, -which(names(data) == "Gross_output_kg")], 2, min_max_normalize)

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

# Train decision tree model
dt_model <- rpart(Gross_output_kg ~ ., data = train_data)

# Make predictions with Decision Tree
dt_predictions_test <- predict(dt_model, newdata = test_data)

# Calculate metrics for Decision Tree
dt_metrics <- calculate_metrics(test_data$Gross_output_kg, dt_predictions_test)

# Train Random Forest model
rf_model <- randomForest(x = train_data[, -which(names(train_data) == "Gross_output_kg")], 
                         y = train_data$Gross_output_kg, 
                         ntree = 500) 

# Make predictions with Random Forest
rf_predictions_test <- predict(rf_model, newdata = test_data[, -which(names(test_data) == "Gross_output_kg")])

# Calculate metrics for Random Forest
rf_metrics <- calculate_metrics(test_data$Gross_output_kg, rf_predictions_test)

# Train SVM model
svm_model <- svm(Gross_output_kg ~ ., data = train_data)

# Make predictions with SVM
svm_predictions_test <- predict(svm_model, newdata = test_data)

# Calculate metrics for SVM
svm_metrics <- calculate_metrics(test_data$Gross_output_kg, svm_predictions_test)

# Train Gradient Boosting Machine (GBM) model
gbm_model <- gbm(Gross_output_kg ~ ., data = train_data, distribution = "gaussian", n.trees = 500, interaction.depth = 6)

# Make predictions with GBM
gbm_predictions_test <- predict(gbm_model, newdata = test_data, n.trees = 500)

# Calculate metrics for GBM
gbm_metrics <- calculate_metrics(test_data$Gross_output_kg, gbm_predictions_test)

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

# Train Multiple Linear Regression model
lm_model <- lm(Gross_output_kg ~ ., data = train_data)

# Make predictions with Multiple Linear Regression
lm_predictions_test <- predict(lm_model, newdata = test_data)

# Calculate metrics for Multiple Linear Regression
lm_metrics <- calculate_metrics(test_data$Gross_output_kg, lm_predictions_test)

# Create a data frame for metrics
models <- c("Decision Tree", "Random Forest", "SVM", "GBM", "XGBoost", "Multiple Linear Regression")
metrics_df <- data.frame(models, t(sapply(list(dt_metrics, rf_metrics, svm_metrics, gbm_metrics, xgb_metrics, lm_metrics), unlist)))

# Print the normalized metrics table
print(metrics_df)

# Melt the metrics data frame for easier plotting
metrics_melted <- melt(metrics_df, id.vars = "models")

# Plot
ggplot(metrics_melted, aes(x = models, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~variable, scales = "free_y") +
  labs(title = "Performance Metrics of Different Models",
       x = "Model",
       y = "Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
