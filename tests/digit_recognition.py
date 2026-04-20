# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import Logistic Regression
from sklearn.linear_model import LogisticRegression

# Import train test split
from sklearn.model_selection import train_test_split

# Import confusion matrix display
from sklearn.metrics import ConfusionMatrixDisplay

# Import digits dataset - handwritten digits 0-9
from sklearn import datasets

# Apply theme
sns.set()

# -------- LOAD DIGITS DATASET --------

# Load handwritten digits - 1797 samples, 8x8 pixels each
digits = datasets.load_digits()

# Print basic info
print("Images shape:", digits.images.shape)
print("Total samples:", len(digits.data))
print("Classes:", digits.target_names)

# -------- SHOW FIRST DIGIT IN NUMBERS --------

print("\nFirst digit in numerical form:")
print(digits.images[0])
print("Label:", digits.target[0])

# -------- SHOW FIRST 10 DIGITS AS IMAGES --------

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    # Show digit image in grayscale
    ax.imshow(digits.images[i], cmap=plt.cm.gray_r)
    # Show label below image
    ax.set_title(f"Label: {digits.target[i]}")
    ax.axis('off')
plt.suptitle('First 10 Handwritten Digits', fontsize=14)
plt.tight_layout()
plt.show()

# -------- CHECK DATASET BALANCE --------

# Plot distribution of all digit classes
plt.xticks([])
plt.hist(digits.target, rwidth=0.9, color='steelblue')
plt.xlabel('Digit Class (0-9)')
plt.ylabel('Count')
plt.title('Dataset Distribution - Is it Balanced?')
plt.xticks(range(10))
plt.show()

# -------- TRAIN LOGISTIC REGRESSION --------

# X = pixel data (64 features per image)
X = digits.data

# y = digit labels (0-9)
y = digits.target

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create and train logistic regression
# max_iter=5000 because multiclass needs more iterations
model = LogisticRegression(max_iter=5000, random_state=0)
model.fit(X_train, y_train)

# Check accuracy
accuracy = model.score(X_test, y_test)
print(f"\nModel Accuracy: {accuracy:.1%}")

# -------- PREDICT SPECIFIC DIGITS --------

print("\n--- Predictions ---")

# Predict digit at index 100
sample_index = 100
predicted = model.predict([digits.data[sample_index]])[0]
actual = digits.target[sample_index]
probabilities = model.predict_proba([digits.data[sample_index]])[0]

print(f"Sample index 100:")
print(f"Actual digit   : {actual}")
print(f"Predicted digit: {predicted}")
print(f"Confidence     : {probabilities[predicted]:.1%}")

# Show that specific digit image
fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(digits.images[sample_index], cmap=plt.cm.gray_r)
ax.set_title(f"Actual: {actual} | Predicted: {predicted}")
ax.axis('off')
plt.show()

# -------- SHOW ALL PROBABILITIES --------

print("\nProbabilities for each digit:")
for digit, prob in enumerate(probabilities):
    bar = "█" * int(prob * 20)
    print(f"Digit {digit}: {prob:.1%} {bar}")

# -------- CONFUSION MATRIX --------

# Plot confusion matrix - shows all 10 classes
fig, ax = plt.subplots(figsize=(10, 8))
ax.grid(False)
ConfusionMatrixDisplay.from_estimator(
    model, X_test, y_test,
    cmap='Blues',
    colorbar=False,
    ax=ax
)
plt.title('Digit Recognition — Confusion Matrix\n(Diagonal = Correct Predictions)')
plt.show()

# -------- SHOW MISCLASSIFIED DIGITS --------

# Find digits that model got wrong
y_predicted = model.predict(X_test)
wrong_indices = np.where(y_predicted != y_test)[0]

print(f"\nTotal mistakes: {len(wrong_indices)} out of {len(y_test)}")

# Show first 5 mistakes
if len(wrong_indices) > 0:
    fig, axes = plt.subplots(1, min(5, len(wrong_indices)),
                              figsize=(12, 3))
    if len(wrong_indices) == 1:
        axes = [axes]
    for i, idx in enumerate(wrong_indices[:5]):
        axes[i].imshow(X_test[idx].reshape(8, 8), cmap=plt.cm.gray_r)
        axes[i].set_title(f"True:{y_test[idx]}\nPred:{y_predicted[idx]}")
        axes[i].axis('off')
    plt.suptitle('Misclassified Digits', fontsize=12)
    plt.tight_layout()
    plt.show()