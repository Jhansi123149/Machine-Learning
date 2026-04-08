# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import Logistic Regression algorithm
from sklearn.linear_model import LogisticRegression

# Import train_test_split to split data
from sklearn.model_selection import train_test_split, cross_val_score

# Import accuracy metrics
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              ConfusionMatrixDisplay, RocCurveDisplay)

# Import famous breast cancer dataset - binary classification!
from sklearn.datasets import load_breast_cancer

# Apply beautiful theme
sns.set()

# -------- LOAD DATA --------

# Load breast cancer dataset
# 569 samples, 30 features, 2 classes (malignant=0, benign=1)
cancer = load_breast_cancer()

# Print dataset info
print("Feature names:", cancer.feature_names[:5], "...")
print("Classes:", cancer.target_names)
print("Total samples:", len(cancer.data))

# X = input features (30 measurements)
X = cancer.data

# y = target (0=malignant/cancer, 1=benign/no cancer)
y = cancer.target

# -------- TRAIN/TEST SPLIT --------

# Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                     test_size=0.2,
                                                     random_state=0)
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# -------- LOGISTIC REGRESSION MODEL --------

# Create logistic regression model
# max_iter=10000 means try hard to find best values
model = LogisticRegression(max_iter=10000, random_state=0)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_predicted = model.predict(X_test)

# -------- ACCURACY METRICS --------

print("\n--- Accuracy Metrics ---")

# Accuracy - overall correct predictions
accuracy = accuracy_score(y_test, y_predicted)
print("Accuracy :", round(accuracy, 3))

# Precision - when model says cancer, how often correct?
precision = precision_score(y_test, y_predicted)
print("Precision:", round(precision, 3))

# Recall - out of all cancer cases, how many did model find?
recall = recall_score(y_test, y_predicted)
print("Recall   :", round(recall, 3))

# F1 Score - balance of precision and recall
f1 = f1_score(y_test, y_predicted)
print("F1 Score :", round(f1, 3))

# Cross validation score
cv_score = cross_val_score(model, X, y, cv=5).mean()
print("CV Score :", round(cv_score, 3))

# -------- PREDICT FOR NEW SAMPLE --------

# Predict probability for first test sample
sample = X_test[0].reshape(1, -1)
predicted_class = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]

print("\n--- Sample Prediction ---")
print("Predicted class:", cancer.target_names[predicted_class])
print(f"Probability malignant: {probabilities[0]:.2%}")
print(f"Probability benign   : {probabilities[1]:.2%}")

# -------- CONFUSION MATRIX --------

# Plot confusion matrix - shows correct and incorrect predictions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_predicted,
    display_labels=cancer.target_names,
    ax=axes[0]
)
axes[0].set_title('Confusion Matrix')

# ROC Curve - shows model performance at different thresholds
RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[1])
axes[1].set_title('ROC Curve')

plt.tight_layout()
plt.show()

# -------- ONE HOT ENCODING DEMO --------

print("\n--- One Hot Encoding Demo ---")

# Create sample data with categorical column
data = [[10, 'red'], [20, 'blue'], [12, 'red'],
        [16, 'green'], [22, 'blue']]
df = pd.DataFrame(data, columns=['Length', 'Color'])

print("Before encoding:")
print(df)

# Apply one-hot encoding using get_dummies
df_encoded = pd.get_dummies(df, columns=['Color'])
print("\nAfter One-Hot encoding:")
print(df_encoded)