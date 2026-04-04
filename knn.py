# Import pandas for data handling
import pandas as pd

# Import matplotlib for graphs
import matplotlib.pyplot as plt

# Import seaborn for styling
import seaborn as sns

# Import KNeighborsClassifier - this is the KNN algorithm
from sklearn.neighbors import KNeighborsClassifier

# Import function to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Import accuracy_score to check how good our model is
from sklearn.metrics import accuracy_score

# Import famous Iris flower dataset (built into sklearn - no CSV needed!)
from sklearn.datasets import load_iris

# Apply beautiful theme
sns.set()

# Load the iris dataset
# This dataset has 150 flowers with 4 measurements each
iris = load_iris()

# Print what features (columns) are in the dataset
print("Features:", iris.feature_names)

# Print what classes (flower types) we are predicting
print("Classes:", iris.target_names)

# X = input data (4 measurements of each flower)
# sepal length, sepal width, petal length, petal width
X = iris.data

# y = answers (which flower type: 0=setosa, 1=versicolor, 2=virginica)
y = iris.target

# Split data: 80% for training, 20% for testing
# test_size=0.2 means 20% goes to testing
# random_state=0 means same split every time
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                     test_size=0.2,
                                                     random_state=0)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create KNN model with n_neighbors=5
# Means: look at 5 nearest neighbors to decide
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model with training data
# This is where model LEARNS from labeled data!
knn.fit(X_train, y_train)

# Predict flower types for test data
predicted = knn.predict(X_test)

# Check accuracy - how many did it get correct?
accuracy = accuracy_score(y_test, predicted)
print("\nModel Accuracy:", accuracy * 100, "%")

# Show actual vs predicted for first 10 flowers
print("\nActual   :", y_test[:10])
print("Predicted:", predicted[:10])

# Plot accuracy for different k values (1 to 20)
# This helps find the best k value!
accuracies = []

for k in range(1, 21):
    # Create KNN with k neighbors
    knn_k = KNeighborsClassifier(n_neighbors=k)
    # Train it
    knn_k.fit(X_train, y_train)
    # Test it
    pred_k = knn_k.predict(X_test)
    # Save accuracy
    accuracies.append(accuracy_score(y_test, pred_k))

# Plot k vs accuracy graph
plt.plot(range(1, 21), accuracies, marker='o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('KNN - Finding Best K Value')
plt.show()