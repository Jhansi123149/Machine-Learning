# Import numpy for data creation
import numpy as np

# Import matplotlib for graphs
import matplotlib.pyplot as plt

# Import seaborn for styling
import seaborn as sns

# Import DecisionTreeRegressor - builds a decision tree
from sklearn.tree import DecisionTreeRegressor

# Import RandomForestRegressor - builds many trees and averages result
from sklearn.ensemble import RandomForestRegressor

# Import GradientBoostingRegressor - builds trees one after another
from sklearn.ensemble import GradientBoostingRegressor

# Import train_test_split to split data into training and testing
from sklearn.model_selection import train_test_split

# Apply beautiful theme
sns.set()

# Create sample data - years of experience vs salary
np.random.seed(0)
X = np.random.uniform(0, 20, 25).reshape(-1, 1)
y = 3984 * X.flatten() + 60040 + np.random.normal(0, 8000, 25)

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                      test_size=0.2,
                                                      random_state=0)

# -------- DECISION TREE --------

# Create decision tree with max depth 3 to prevent overfitting
dt_model = DecisionTreeRegressor(max_depth=3, random_state=0)

# Train the model
dt_model.fit(X_train, y_train)

# Predict salary for 10 years experience
dt_pred = dt_model.predict([[10]])
print("Decision Tree prediction for 10 years: $", round(dt_pred[0], 2))

# Check accuracy score
dt_score = dt_model.score(X_test, y_test)
print("Decision Tree accuracy:", round(dt_score, 3))

# -------- RANDOM FOREST --------

# Create random forest with 100 trees
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)

# Train the model
rf_model.fit(X_train, y_train)

# Predict salary for 10 years experience
rf_pred = rf_model.predict([[10]])
print("\nRandom Forest prediction for 10 years: $", round(rf_pred[0], 2))

# Check accuracy score
rf_score = rf_model.score(X_test, y_test)
print("Random Forest accuracy:", round(rf_score, 3))

# -------- GRADIENT BOOSTING --------

# Create GBM with 100 trees and learning rate 0.1
gb_model = GradientBoostingRegressor(n_estimators=100,
                                      learning_rate=0.1,
                                      random_state=0)

# Train the model
gb_model.fit(X_train, y_train)

# Predict salary for 10 years experience
gb_pred = gb_model.predict([[10]])
print("\nGradient Boosting prediction for 10 years: $", round(gb_pred[0], 2))

# Check accuracy score
gb_score = gb_model.score(X_test, y_test)
print("Gradient Boosting accuracy:", round(gb_score, 3))

# -------- COMPARE ALL MODELS --------

# Plot all 3 models together to compare
X_plot = np.linspace(0, 20, 100).reshape(-1, 1)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, s=50, label='Actual data', zorder=5)
plt.plot(X_plot, dt_model.predict(X_plot), label='Decision Tree', color='red')
plt.plot(X_plot, rf_model.predict(X_plot), label='Random Forest', color='green')
plt.plot(X_plot, gb_model.predict(X_plot), label='Gradient Boosting', color='blue')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.title('Decision Tree vs Random Forest vs Gradient Boosting')
plt.legend()
plt.show()

# -------- ACCURACY COMPARISON --------

# Print all model scores together for easy comparison
print("\n--- Model Comparison ---")
print("Decision Tree score  :", round(dt_score, 3))
print("Random Forest score  :", round(rf_score, 3))
print("Gradient Boosting score:", round(gb_score, 3))