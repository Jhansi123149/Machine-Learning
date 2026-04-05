# Import pandas and seaborn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()

# Convert to pandas dataframe for easy handling
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add class column (flower type)
df['class'] = iris.target

# Create pair plot - shows relationship between all columns!
# This is one of the most useful visualization tools in ML!
sns.pairplot(df, hue='class')
plt.suptitle('Iris Dataset - Pair Plot', y=1.02)
plt.show()