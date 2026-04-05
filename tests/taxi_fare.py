# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from math import sqrt

# Apply beautiful theme
sns.set()

# -------- CREATE SAMPLE TAXI DATA --------
# We create fake taxi data since real dataset is very large
# In real project you would use: pd.read_csv('Data/taxi-fares.csv')

np.random.seed(0)
n = 500

# Create random pickup and dropoff locations in NYC area
pickup_lat = np.random.uniform(40.63, 40.85, n)
pickup_lon = np.random.uniform(-74.03, -73.75, n)
dropoff_lat = np.random.uniform(40.63, 40.85, n)
dropoff_lon = np.random.uniform(-74.03, -73.75, n)

# Calculate distance in miles using coordinates
x = (dropoff_lon - pickup_lon) * 54.6
y = (dropoff_lat - pickup_lat) * 69.0
distance = np.sqrt(x**2 + y**2)

# Create day of week (0=Monday to 6=Sunday)
day_of_week = np.random.randint(0, 7, n)

# Create pickup time (0 to 23 hours)
pickup_time = np.random.randint(0, 24, n)

# Create fare amount based on distance + some randomness
# Base fare $2.5 + $2.5 per mile + time factor
fare_amount = 2.5 + (distance * 2.5) + (pickup_time * 0.1) + np.random.normal(0, 2, n)

# Create dataframe
df = pd.DataFrame({
    'fare_amount': fare_amount,
    'day_of_week': day_of_week,
    'pickup_time': pickup_time,
    'distance': distance
})

# Remove outliers - keep realistic fare amounts and distances
df = df[(df['distance'] > 0.1) & (df['distance'] < 10.0)]
df = df[(df['fare_amount'] > 0.0) & (df['fare_amount'] < 50.0)]

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Check correlation between columns and fare_amount
print("\nCorrelation with fare_amount:")
corr_matrix = df.corr()
print(corr_matrix['fare_amount'].sort_values(ascending=False))

# Plot distance vs fare amount
plt.scatter(df['distance'], df['fare_amount'], alpha=0.3, s=10)
plt.xlabel('Distance (miles)')
plt.ylabel('Fare Amount ($)')
plt.title('Distance vs Fare Amount')
plt.show()

# -------- PREPARE DATA FOR TRAINING --------

# X = input features (distance, day_of_week, pickup_time)
X = df.drop(['fare_amount'], axis=1)

# y = target (fare_amount to predict)
y = df['fare_amount']

print("\nFeatures used:", list(X.columns))

# -------- CROSS VALIDATION - compare 3 models --------

print("\n--- Cross Validation Scores (5-fold) ---")

# Linear Regression with cross validation
lr_model = LinearRegression()
lr_score = cross_val_score(lr_model, X, y, cv=5).mean()
print("Linear Regression R² score:", round(lr_score, 3))

# Random Forest with cross validation
rf_model = RandomForestRegressor(random_state=0)
rf_score = cross_val_score(rf_model, X, y, cv=5).mean()
print("Random Forest R² score    :", round(rf_score, 3))

# Gradient Boosting with cross validation
gb_model = GradientBoostingRegressor(random_state=0)
gb_score = cross_val_score(gb_model, X, y, cv=5).mean()
print("Gradient Boosting R² score:", round(gb_score, 3))

# -------- TRAIN BEST MODEL AND PREDICT --------

# Find best model
scores = {'Linear Regression': lr_score,
          'Random Forest': rf_score,
          'Gradient Boosting': gb_score}
best_model_name = max(scores, key=scores.get)
print("\nBest model:", best_model_name)

# Train gradient boosting on full data
gb_model.fit(X, y)

# Predict fare for Friday 5PM, 2 mile trip
friday_trip = pd.DataFrame({
    'day_of_week': [4],    # 4 = Friday
    'pickup_time': [17],   # 17 = 5:00 PM
    'distance': [2.0]      # 2 miles
})
friday_fare = gb_model.predict(friday_trip)
print("\nPredicted fare - Friday 5PM, 2 miles: $", round(friday_fare[0], 2))

# Predict fare for Saturday 5PM, same trip
saturday_trip = pd.DataFrame({
    'day_of_week': [5],    # 5 = Saturday
    'pickup_time': [17],   # 17 = 5:00 PM
    'distance': [2.0]      # 2 miles
})
saturday_fare = gb_model.predict(saturday_trip)
print("Predicted fare - Saturday 5PM, 2 miles: $", round(saturday_fare[0], 2))

# -------- PLOT MODEL COMPARISON --------

model_names = ['Linear\nRegression', 'Random\nForest', 'Gradient\nBoosting']
model_scores = [lr_score, rf_score, gb_score]

plt.bar(model_names, model_scores, color=['blue', 'green', 'red'])
plt.ylabel('R² Score')
plt.title('Model Comparison - Cross Validation Scores')
plt.ylim(0, 1)
plt.show()