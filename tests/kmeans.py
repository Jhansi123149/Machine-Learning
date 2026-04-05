# Import library for drawing graphs, using 'plt' as shortcut name
import matplotlib.pyplot as plt

# Import library for making graphs look beautiful, 'sns' as shortcut
import seaborn as sns

# Import function to create fake data for practice
from sklearn.datasets import make_blobs

# Import KMeans algorithm - this is the brain of the program!
from sklearn.cluster import KMeans

# Apply default beautiful theme to graph
sns.set()

# Create 300 fake points arranged near 4 groups
# cluster_std=0.8 means points should be close together
# random_state=0 means same output every time we run
points, cluster_indexes = make_blobs(n_samples=300, centers=4,
                                      cluster_std=0.8, random_state=0)

# Get horizontal position of every point
x = points[:, 0]

# Get vertical position of every point
y = points[:, 1]

# Tell KMeans we want 4 groups
kmeans = KMeans(n_clusters=4, random_state=0)

# Give all points to KMeans and let it find the groups - magic happens here!
kmeans.fit(points)

# Ask KMeans which group each point belongs to
predicted = kmeans.predict(points)

# Plot points on graph, each group gets different color
# c=predicted means color by group, s=50 means point size, alpha=0.7 means slightly transparent
plt.scatter(x, y, c=predicted, s=50, alpha=0.7, cmap='viridis')

# Get the center position of all 4 groups
centers = kmeans.cluster_centers_

# Plot the 4 centers as red stars on graph
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='*')

# Add title to the graph
plt.title("K-Means Clustering!")

# Display the graph on screen
plt.show()