# Import numpy for creating sample data
import numpy as np

# Import matplotlib for graphs
import matplotlib.pyplot as plt

# Import seaborn for styling
import seaborn as sns

# Import KNeighborsRegressor - KNN for predicting numbers
from sklearn.neighbors import KNeighborsRegressor

# Import train_test_split to split data into training and testing
from sklearn.model_selection import train_test_split

# Apply beautiful theme
sns.set()

# Create sample programmer data
# X = years of experience (0 to 20 years)
np.random.seed(0)
X = np.random.uniform(0, 20, 20).reshape(-1, 1)

# y = salary in dollars (more experience = more salary roughly)
y = 50000 + (X.flatten() * 5000) + np.random.normal(0, 10000, 20)

# Plot raw data to see the relationship
plt.scatter(X, y, s=50)
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Raw Data - Experience vs Salary')
plt.show()

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                     test_size=0.2,
                                                     random_state=0)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Try KNN Regression with n=5 neighbors
knn5 = KNeighborsRegressor(n_neighbors=5)

# Train the model with training data
knn5.fit(X_train, y_train)

# Predict salary for 10 years experience
prediction_5 = knn5.predict([[10]])
print("\nn=5 prediction for 10 years experience: $", round(prediction_5[0], 2))

# Try KNN Regression with n=10 neighbors
knn10 = KNeighborsRegressor(n_neighbors=10)
knn10.fit(X_train, y_train)
prediction_10 = knn10.predict([[10]])
print("n=10 prediction for 10 years experience: $", round(prediction_10[0], 2))

# Plot both models to compare
X_plot = np.linspace(0, 20, 100).reshape(-1, 1)

plt.scatter(X, y, s=50, label='Actual data')
plt.plot(X_plot, knn5.predict(X_plot), label='n=5', color='red')
plt.plot(X_plot, knn10.predict(X_plot), label='n=10', color='green')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('KNN Regression - n=5 vs n=10')
plt.legend()
plt.show()

# Check model accuracy using score method
# Score closer to 1.0 means better model!
score_5 = knn5.score(X_test, y_test)
score_10 = knn10.score(X_test, y_test)
print("\nn=5 model score:", round(score_5, 3))
print("n=10 model score:", round(score_10, 3))