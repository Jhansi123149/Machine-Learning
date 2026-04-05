# Import numpy for creating data
import numpy as np

# Import matplotlib for graphs
import matplotlib.pyplot as plt

# Import seaborn for styling
import seaborn as sns

# Import LinearRegression algorithm
from sklearn.linear_model import LinearRegression

# Import train_test_split to split data
from sklearn.model_selection import train_test_split

# Apply beautiful theme
sns.set()

# Create sample data - years of experience vs salary
np.random.seed(0)

# X = years of experience (0 to 20)
X = np.random.uniform(0, 20, 25).reshape(-1, 1)

# y = salary (more experience = more salary)
y = 3984 * X.flatten() + 60040 + np.random.normal(0, 8000, 25)

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                     test_size=0.2,
                                                     random_state=0)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create Linear Regression model
model = LinearRegression()

# Train the model - it finds best m and b values!
model.fit(X_train, y_train)

# Print the m (slope) and b (intercept) values
print("\nSlope (m):", round(model.coef_[0], 2))
print("Intercept (b):", round(model.intercept_, 2))
print("Equation: y =", round(model.coef_[0], 2), "* x +", round(model.intercept_, 2))

# Predict salary for 10 years experience
prediction = model.predict([[10]])
print("\nPredicted salary for 10 years experience: $", round(prediction[0], 2))

# Check model accuracy
score = model.score(X_test, y_test)
print("Model accuracy score:", round(score, 3))

# Plot the data and regression line
plt.scatter(X, y, s=50, label='Actual data')

# Draw the regression line
X_line = np.linspace(0, 20, 100).reshape(-1, 1)
plt.plot(X_line, model.predict(X_line), color='red', label='Regression line')

# Mark the prediction point for 10 years
plt.scatter([10], prediction, color='green', s=200, marker='*',
            label=f'Prediction: ${round(prediction[0])}')

plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.title('Linear Regression - Experience vs Salary')
plt.legend()
plt.show()