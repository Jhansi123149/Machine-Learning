# Import pandas library for reading CSV files and working with data tables
import pandas as pd

# Import matplotlib for drawing graphs
import matplotlib.pyplot as plt

# Import seaborn for beautiful graph styling
import seaborn as sns

# Import KMeans algorithm
from sklearn.cluster import KMeans

# Apply beautiful theme to graphs
sns.set()

# Read the CSV file and load customer data into a table called 'customers'
customers = pd.read_csv('Data/customers.csv')

# Show first 5 rows of the data to understand what it looks like
print(customers.head())

# Take only Annual Income and Spending Score columns for clustering
# iloc[:, 3:5] means - all rows, columns 3 and 4 only
points = customers.iloc[:, 3:5].values

# Get horizontal position (Annual Income) of every point
x = points[:, 0]

# Get vertical position (Spending Score) of every point
y = points[:, 1]

# Plot raw data first to see how it looks without clustering
plt.scatter(x, y, s=50, alpha=0.7)

# Label the horizontal axis
plt.xlabel('Annual Income (k$)')

# Label the vertical axis
plt.ylabel('Spending Score')

# Display the raw data graph
plt.show()


# -------- ELBOW METHOD --------

# Empty list to store inertia values
inertias = []

# Try clusters from 1 to 9 and record inertia each time
for i in range(1, 10):
    # Create KMeans with i clusters
    kmeans = KMeans(n_clusters=i, random_state=0)
    # Fit the data
    kmeans.fit(points)
    # Save inertia to list
    inertias.append(kmeans.inertia_)

# Plot elbow curve - look for the bend/elbow in the line!
plt.plot(range(1, 10), inertias)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method - Find best cluster count')
plt.show()

# -------- FINAL CLUSTERING WITH 5 GROUPS --------

# Apply KMeans with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(points)
predicted_cluster_indexes = kmeans.predict(points)

# Plot with 5 different colors
plt.scatter(x, y, c=predicted_cluster_indexes, s=50, alpha=0.7, cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')

# Plot centroids as red dots
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100)
plt.title('Customer Segments - 5 Groups!')
plt.show()

# Add Cluster column to customer data
df = customers.copy()
df['Cluster'] = kmeans.predict(points)
print(df.head())