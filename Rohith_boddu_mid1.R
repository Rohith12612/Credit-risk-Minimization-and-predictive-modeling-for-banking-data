

# Let's start by reading in the training and test data
train_data <- read.csv("E:/2025/March/25-03/R Project/training.csv")
test_data <- read.csv("E:/2025/March/25-03/R Project/test.csv")

# I want to inspect the first few rows of the training data
head(train_data)

# Now I will check the names of the variables to make sure they match the assignment description
names(test_data)

# Let's also check the structure of the training dataset
str(train_data)

# Let's efine a function to convert discrete variables to factors using confirmed column names
to_factor <- function(df) {
  df$Account.Balance <- as.factor(df$Account.Balance)
  df$Payment.Status.of.Previous.Credit <- as.factor(df$Payment.Status.of.Previous.Credit)
  df$Purpose <- as.factor(df$Purpose)
  df$Value.Savings.Stocks <- as.factor(df$Value.Savings.Stocks)
  df$Length.of.current.employment <- as.factor(df$Length.of.current.employment)
  df$Sex...Marital.Status <- as.factor(df$Sex...Marital.Status)
  df$Guarantors <- as.factor(df$Guarantors)
  df$Most.valuable.available.asset <- as.factor(df$Most.valuable.available.asset)
  df$Concurrent.Credits <- as.factor(df$Concurrent.Credits)
  df$Type.of.apartment <- as.factor(df$Type.of.apartment)
  df$Occupation <- as.factor(df$Occupation)
  df$Telephone <- as.factor(df$Telephone)
  df$Foreign.Worker <- as.factor(df$Foreign.Worker)
  df$No.of.Credits.at.this.Bank <- as.factor(df$No.of.Credits.at.this.Bank)
  df$No.of.dependents <- as.factor(df$No.of.dependents)
  
  # Convert the response variable to factor (if it exists in the dataset)
  if ("Creditability" %in% names(df)) {
    df$Creditability <- as.factor(df$Creditability)
  }
  
  return(df)
}

# Now we apply the factor conversion to both training and test datasets
train_data <- to_factor(train_data)
test_data <- to_factor(test_data)

# Let's confirm the final structure of all variables
str(train_data)

# Now I will create a summary table showing the class of each variable
# This will help me later when describing them in the Word report

summary_table <- sapply(train_data, class)
summary_table

########################################################
# Section 2: Basic Models
## Goal: Train and evaluate classification models using train_data
##       and generate predictions using test_data
########################################################

########################################################
## 2.1 Logistic Regression
########################################################


# Fit the logistic regression model using all main effects
model_logit <- glm(Creditability ~ ., data = train_data, family = binomial)

# Predict probabilities on training data (for evaluation)
prob_logit <- predict(model_logit, newdata = train_data, type = "response")

# Convert probabilities to class predictions using 0.5 threshold
pred_logit <- ifelse(prob_logit > 0.5, "1", "0")
pred_logit <- as.factor(pred_logit)

# Confusion matrix
confusion_logit <- table(Predicted = pred_logit, Actual = train_data$Creditability)
print(confusion_logit)

# Accuracy
accuracy_logit <- mean(pred_logit == train_data$Creditability)
print(accuracy_logit)

# Check model summary for assumptions (significant predictors)
summary(model_logit)


# 2.2 NAIVE BAYES
########################################################

library(e1071)

# Train model
model_nb <- naiveBayes(Creditability ~ ., data = train_data)

# Predict on training data
pred_nb <- predict(model_nb, newdata = train_data)

# Confusion matrix and accuracy
confusion_nb <- table(Predicted = pred_nb, Actual = train_data$Creditability)
print(confusion_nb)

accuracy_nb <- mean(pred_nb == train_data$Creditability)
print(accuracy_nb)

# 2.3 LINEAR DISCRIMINANT ANALYSIS (LDA)
########################################################

library(MASS)

# Train LDA model
model_lda <- lda(Creditability ~ ., data = train_data)

# Summary of the model
print(model_lda)

# Predict on training data
pred_lda <- predict(model_lda, newdata = train_data)$class

# Confusion matrix and accuracy
confusion_lda <- table(Predicted = pred_lda, Actual = train_data$Creditability)
print(confusion_lda)

accuracy_lda <- mean(pred_lda == train_data$Creditability)
print(accuracy_lda)

# 2.4 QUADRATIC DISCRIMINANT ANALYSIS (QDA)
########################################################

library(MASS)

# Train QDA model
model_qda <- qda(Creditability ~ ., data = train_data)

# Summary of the QDA model
print(model_qda)

# Predict on training data
pred_qda <- predict(model_qda, newdata = train_data)$class

# Confusion matrix and accuracy
confusion_qda <- table(Predicted = pred_qda, Actual = train_data$Creditability)
print(confusion_qda)

accuracy_qda <- mean(pred_qda == train_data$Creditability)
print(accuracy_qda)


# 2.5 K-NEAREST NEIGHBORS (KNN)
########################################################

library(class)

# Select and scale only numeric variables
numeric_vars <- c("Duration.of.Credit..month.", "Credit.Amount", 
                  "Instalment.per.cent", "Duration.in.Current.address", "Age..years.")

train_X <- scale(train_data[, numeric_vars])
test_X <- scale(test_data[, numeric_vars], 
                center = attr(train_X, "scaled:center"), 
                scale = attr(train_X, "scaled:scale"))

train_Y <- train_data$Creditability

# Fit KNN (k = 5) and predict on training set for evaluation
pred_knn <- knn(train = train_X, test = train_X, cl = train_Y, k = 5)

# Confusion matrix and accuracy
confusion_knn <- table(Predicted = pred_knn, Actual = train_Y)
print(confusion_knn)

accuracy_knn <- mean(pred_knn == train_Y)
print(accuracy_knn)


########################################################
## 2.summary - Compare Models (Table + Visualization)
########################################################

# Combine all accuracy values into a named vector
model_accuracies <- c(
  Logistic = accuracy_logit,
  NaiveBayes = accuracy_nb,
  LDA = accuracy_lda,
  QDA = accuracy_qda,
  KNN = accuracy_knn
)

# Print model comparison table
accuracy_table <- data.frame(
  Model = names(model_accuracies),
  Accuracy = round(model_accuracies, 4)
)
print(accuracy_table)



# Plot model comparison
barplot(
  model_accuracies,
  main = "Model Accuracy Comparison",
  col = "lightblue",
  ylim = c(0, 1),
  ylab = "Accuracy",
  xlab = "Model",
  las = 2
)
abline(h = max(model_accuracies), col = "red", lty = 2)
text(x = 1:5, y = model_accuracies, 
     labels = round(model_accuracies, 3), pos = 3, cex = 0.9)


########################################################
## 3.1 Variable Selection - Stepwise Logistic Regression
## Goal: Identify most relevant predictors using stepwise AIC
########################################################

# Start with full logistic regression model using all variables
full_model <- glm(Creditability ~ ., data = train_data, family = binomial)

# Run stepwise selection using both directions (backward and forward)
stepwise_model <- step(full_model, direction = "both", trace = FALSE)

# View summary of the selected model
summary(stepwise_model)


########################################################
## 3.2 Principal Component Analysis (PCA)
## Goal: Reduce dimensionality and determine optimal number of components
########################################################

# Select only numeric variables for PCA
numeric_vars <- c("Duration.of.Credit..month.", "Credit.Amount",
                  "Instalment.per.cent", "Duration.in.Current.address", "Age..years.")

# Standardize numeric variables before PCA
scaled_train <- scale(train_data[, numeric_vars])

# Perform PCA
pca_result <- prcomp(scaled_train, center = TRUE, scale. = TRUE)

# Summary of PCA: shows standard deviation, proportion of variance, cumulative variance
summary(pca_result)


# Scree plot of variance explained
plot(pca_result, type = "l", main = "Scree Plot: Variance Explained by PCs")

# Cumulative variance explained
cum_var <- cumsum(pca_result$sdev^2 / sum(pca_result$sdev^2))
plot(cum_var, xlab = "Number of Principal Components", ylab = "Cumulative Proportion of Variance Explained",
     type = "b", pch = 19, col = "blue", main = "Cumulative Variance Explained")


########################################################
## 3.3 Cross-validation
## Goal: Evaluate stability of stepwise variable selection and PCA results
########################################################

set.seed(123)  # For reproducibility

# Create fold assignments
folds <- cut(seq(1, nrow(train_data)), breaks = 3, labels = FALSE)

# Store results
stepwise_vars_list <- list()
pca_cumvar_list <- list()
pca_ncomp_list <- c()

# Define numeric variables used for PCA
numeric_vars <- c("Duration.of.Credit..month.", "Credit.Amount",
                  "Instalment.per.cent", "Duration.in.Current.address", "Age..years.")

for (i in 1:3) {
  cat("Fold", i, "\n")
  
  # Split data
  val_indices <- which(folds == i)
  train_fold <- train_data[-val_indices, ]
  val_fold <- train_data[val_indices, ]
  
  ## STEPWISE LOGISTIC REGRESSION (suppress all warnings)
  suppressWarnings({
    step_model <- glm(Creditability ~ ., data = train_fold, family = binomial)
    stepwise_fit <- step(step_model, direction = "both", trace = FALSE)
  })
  
  # Store selected variables
  stepwise_vars_list[[i]] <- names(coef(stepwise_fit))[-1]  # Exclude intercept
  
  ## PCA ON NUMERIC VARIABLES
  scaled_fold <- scale(train_fold[, numeric_vars])
  pca_fold <- prcomp(scaled_fold, center = TRUE, scale. = TRUE)
  
  # Cumulative variance explained
  cumvar <- cumsum(pca_fold$sdev^2 / sum(pca_fold$sdev^2))
  pca_cumvar_list[[i]] <- cumvar
  
  # Determine number of components to retain (≥ 80% variance)
  n_comp <- which(cumvar >= 0.80)[1]
  pca_ncomp_list[i] <- n_comp
}

# Summary of variable selection across folds
cat("\nSelected variables per fold:\n")
for (i in 1:3) {
  cat(paste("Fold", i, ":"), paste(stepwise_vars_list[[i]], collapse = ", "), "\n")
}

# Table of variable frequencies
all_selected_vars <- unlist(stepwise_vars_list)
selected_freq <- sort(table(all_selected_vars), decreasing = TRUE)
cat("\nVariable selection frequency across folds:\n")
print(selected_freq)

# PCA component count consistency
cat("\nNumber of principal components retained per fold (≥80% variance):\n")
print(pca_ncomp_list)

# Let's plot a barplot of variable selection frequencies
barplot(
  selected_freq,
  main = "Variable Selection Frequency Across 3 Folds",
  col = "steelblue",
  ylab = "Frequency",
  xlab = "Variables",
  las = 2,         # Rotate x labels for readability
  cex.names = 0.7  # Shrink label font
)
abline(h = 3, col = "red", lty = 2)  # Highlight variables selected in all folds


# Barplot of PCA components retained per fold
barplot(
  pca_ncomp_list,
  names.arg = paste("Fold", 1:3),
  main = "Number of Principal Components Retained per Fold",
  col = "darkorange",
  ylab = "Number of Components",
  ylim = c(0, 5)
)
abline(h = 3, col = "blue", lty = 2)
abline(h = 4, col = "blue", lty = 2)

########################################################
## 4. Best Guesses – Final Model Selection and Prediction
## Final Model: Quadratic Discriminant Analysis (QDA)
########################################################

# Load required library
library(MASS)

# Train final QDA model on the full training dataset
final_model_qda <- qda(Creditability ~ ., data = train_data)

# ----------------------
# Training Set Evaluation
# ----------------------

# Predict on training data
train_pred <- predict(final_model_qda, newdata = train_data)
train_class <- train_pred$class
train_probs <- train_pred$posterior

# Confusion matrix and accuracy
conf_matrix <- table(Predicted = train_class, Actual = train_data$Creditability)
accuracy <- mean(train_class == train_data$Creditability)

cat("Confusion Matrix (Training Set):\n")
print(conf_matrix)
cat("\nAccuracy:", round(accuracy, 4), "\n")

# ----------------------
# ROC Curve and AUC
# ----------------------

library(ROCR)

# Convert factor labels to numeric (required for ROCR)
true_labels <- as.numeric(train_data$Creditability) - 1
pred_scores <- train_probs[, 2]  # Probability of class "1"

# Create prediction and performance objects
pred_obj <- prediction(pred_scores, true_labels)
perf_obj <- performance(pred_obj, "tpr", "fpr")

# Plot ROC Curve
plot(perf_obj, col = "darkgreen", lwd = 2, main = "ROC Curve for QDA (Training Set)")
abline(0, 1, lty = 2, col = "gray")

# Compute AUC
auc <- performance(pred_obj, "auc")@y.values[[1]]
cat("AUC:", round(auc, 4), "\n")

# ----------------------
# Test Set Prediction
# ----------------------

# Predict on test data
test_pred <- predict(final_model_qda, newdata = test_data)$class

# Save predictions
write.csv(test_pred, "E:/2025/March/25-03/R Project/Rohith_boddu_mid1.csv", row.names = FALSE)

# Save R workspace
save.image("E:/2025/March/25-03/R Project/Rohith_boddu_mid1.RData")

