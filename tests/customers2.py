# Import pandas for data handling
import pandas as pd

# Import matplotlib for graphs
import matplotlib.pyplot as plt

# Import seaborn for styling
import seaborn as sns

# Import KMeans algorithm
from sklearn.cluster import KMeans

# Import LabelEncoder to convert text (Male/Female) to numbers (0/1)
from sklearn.preprocessing import LabelEncoder

# Apply beautiful theme
sns.set()

# Read customer data from CSV file
customers = pd.read_csv('Data/customers.csv')

# Create a copy of customers data to work with
df = customers.copy()

# Create LabelEncoder object - this converts text to numbers
encoder = LabelEncoder()

# Convert Gender column: Male=0, Female=1
df['Gender'] = encoder.fit_transform(df['Gender'])

# Show first 5 rows to confirm Gender is now 0 and 1
print("After encoding:")
print(df.head())

# Take Gender, Age, Annual Income, Spending Score columns for clustering
# iloc[:, 1:5] means all rows, columns 1 to 4
points = df.iloc[:, 1:5].values

# Find best number of clusters using Elbow Method
inertias = []

# Try 1 to 9 clusters and record inertia
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(points)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.plot(range(1, 10), inertias)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method - 4 Dimensions')
plt.show()

# Apply KMeans with 5 clusters (elbow shows 5 is best)
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(points)

# Add Cluster column to dataframe
df['Cluster'] = kmeans.predict(points)

# Show cluster summary - average age, income, spending per cluster
results = pd.DataFrame(kmeans.cluster_centers_,
                       columns=['Gender', 'Age',
                                'Annual Income', 'Spending Score'])
print("\nCluster Centers (averages):")
print(results)

# Show first 5 rows with cluster assigned
print("\nCustomers with Cluster:")
print(df.head())