# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import machine learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Import model evaluation tools
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              ConfusionMatrixDisplay)

# Apply beautiful theme
sns.set()

# ======== PART 1: TITANIC SURVIVAL PREDICTION ========

print("=" * 50)
print("PART 1: TITANIC SURVIVAL PREDICTION")
print("=" * 50)

# Create sample Titanic-like data
# In real project: pd.read_csv('Data/titanic.csv')
np.random.seed(0)
n = 500

# Create sample passenger data
age = np.random.uniform(1, 80, n)
pclass = np.random.choice([1, 2, 3], n, p=[0.3, 0.3, 0.4])
sex = np.random.choice([0, 1], n)  # 0=male, 1=female

# Survival probability based on real Titanic patterns
# Women and 1st class passengers had higher survival rates
survival_prob = (0.3 +
                 (sex * 0.4) +           # women more likely to survive
                 ((4 - pclass) * 0.1) +  # 1st class more likely
                 (age < 15) * 0.2)       # children more likely

survival_prob = np.clip(survival_prob, 0, 1)
survived = (np.random.random(n) < survival_prob).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'Survived': survived,
    'Age': age,
    'Sex_female': sex,
    'Sex_male': 1 - sex,
    'Pclass_1': (pclass == 1).astype(int),
    'Pclass_2': (pclass == 2).astype(int),
    'Pclass_3': (pclass == 3).astype(int)
})

print("\nDataset shape:", df.shape)
print("\nSurvival rate:", round(df['Survived'].mean() * 100, 1), "%")
print("\nFirst 5 rows:")
print(df.head())

# Prepare features and target
X = df.drop(['Survived'], axis=1)
y = df['Survived']

# Split data with stratify - keeps same ratio in train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0)

# Train Logistic Regression
model = LogisticRegression(random_state=0, max_iter=1000)
model.fit(X_train, y_train)

# Cross validation score
cv_score = cross_val_score(model, X, y, cv=5).mean()
print("\nCross Validation Score:", round(cv_score, 3))

# Confusion Matrix
y_pred = model.predict(X_test)
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred), 3))
print("Recall:", round(recall_score(y_test, y_pred), 3))

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(6, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=['Perished', 'Survived'],
    ax=ax, cmap='Blues'
)
plt.title('Titanic - Confusion Matrix')
plt.show()

# Predict for specific passengers
print("\n--- Passenger Predictions ---")

# 30-year-old female, 1st class
female_1st = pd.DataFrame({
    'Age': [30], 'Sex_female': [1], 'Sex_male': [0],
    'Pclass_1': [1], 'Pclass_2': [0], 'Pclass_3': [0]
})
prob_female = model.predict_proba(female_1st)[0][1]
print(f"30yr Female 1st class survival: {prob_female:.1%}")

# 60-year-old male, 3rd class
male_3rd = pd.DataFrame({
    'Age': [60], 'Sex_female': [0], 'Sex_male': [1],
    'Pclass_1': [0], 'Pclass_2': [0], 'Pclass_3': [1]
})
prob_male = model.predict_proba(male_3rd)[0][1]
print(f"60yr Male 3rd class survival  : {prob_male:.1%}")

# ======== PART 2: DIGIT RECOGNITION ========

print("\n" + "=" * 50)
print("PART 2: DIGIT RECOGNITION (Multiclass)")
print("=" * 50)

# Import digits dataset
from sklearn.datasets import load_digits

# Load handwritten digits dataset
digits = load_digits()
print("\nImages shape:", digits.images.shape)
print("Target shape:", digits.target.shape)
print("Classes:", digits.target_names)

# Show first 10 digits
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flatten()):
    # Show digit image
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f"Label: {digits.target[i]}")
    ax.axis('off')
plt.suptitle('First 10 Handwritten Digits')
plt.tight_layout()
plt.show()

# Prepare data
X_digits = digits.data    # 8x8 pixels flattened to 64 features
y_digits = digits.target  # 0 to 9

# Split data
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_digits, y_digits, test_size=0.2, random_state=0)

# Train Logistic Regression for multiclass
lr_digits = LogisticRegression(max_iter=10000, random_state=0)
lr_digits.fit(X_train_d, y_train_d)

# Accuracy
digit_accuracy = lr_digits.score(X_test_d, y_test_d)
print("\nDigit Recognition Accuracy:", round(digit_accuracy * 100, 2), "%")

# Cross validation
cv_digits = cross_val_score(lr_digits, X_digits, y_digits, cv=5).mean()
print("Cross Validation Score:", round(cv_digits, 3))

# Predict a specific digit
sample_digit = X_test_d[0].reshape(1, -1)
predicted_digit = lr_digits.predict(sample_digit)[0]
actual_digit = y_test_d[0]
probabilities = lr_digits.predict_proba(sample_digit)[0]

print(f"\nSample prediction:")
print(f"Actual digit   : {actual_digit}")
print(f"Predicted digit: {predicted_digit}")
print(f"Confidence     : {probabilities[predicted_digit]:.1%}")

# Show confusion matrix for digits
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test_d,
    lr_digits.predict(X_test_d),
    ax=ax, cmap='Blues'
)
plt.title('Digit Recognition - Confusion Matrix')
plt.show()