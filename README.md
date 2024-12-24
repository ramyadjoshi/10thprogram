# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the Wisconsin Breast Cancer dataset
data = load_breast_cancer()
X = data.data  # Features
y = data.target  # Target variable (0 = malignant, 1 = benign)
feature_names = data.feature_names
target_names = data.target_names

# Standardize the data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
X_pca = pca.fit_transform(X_scaled)

# Get explained variance ratio for each component
explained_variance_ratio = pca.explained_variance_ratio_

# Create a DataFrame for visualization
pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
pca_df['Target'] = y

# Plot the PCA results
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Target', palette='Set1', alpha=0.8)
plt.title('PCA of Wisconsin Breast Cancer Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(target_names)
plt.grid()
plt.show()

# Plot explained variance ratio
plt.figure(figsize=(8, 5))
plt.bar(range(1, 3), explained_variance_ratio, tick_label=['PCA1', 'PCA2'], color='skyblue')
plt.title('Explained Variance Ratio of PCA Components')
plt.xlabel('Principal Components')
plt.ylabel('Variance Explained')
plt.show()

# Full PCA with all components for analysis
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Plot cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='b')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained')
plt.grid()
plt.show()

# Print key insights
print("PCA Analysis of Wisconsin Breast Cancer Dataset")
print("-------------------------------------------------")
print(f"Explained Variance (PCA1): {explained_variance_ratio[0]:.4f}")
print(f"Explained Variance (PCA2): {explained_variance_ratio[1]:.4f}")
print("Cumulative Variance Explained by All Components:")
for i, cum_var in enumerate(cumulative_variance, start=1):
    print(f"  Component {i}: {cum_var:.4f}")

PCA Analysis of Wisconsin Breast Cancer Dataset
-------------------------------------------------
Explained Variance (PCA1): 0.4427
Explained Variance (PCA2): 0.1897
Cumulative Variance Explained by All Components:
  Component 1: 0.4427
  Component 2: 0.6324
  Component 3: 0.7264
  Component 4: 0.7924
  Component 5: 0.8473
  Component 6: 0.8876
  Component 7: 0.9101
  Component 8: 0.9260
  Component 9: 0.9399
  Component 10: 0.9516
  Component 11: 0.9614
  Component 12: 0.9701
  Component 13: 0.9781
  Component 14: 0.9834
  Component 15: 0.9865
  Component 16: 0.9892
  Component 17: 0.9911
  Component 18: 0.9929
  Component 19: 0.9945
  Component 20: 0.9956
  Component 21: 0.9966
  Component 22: 0.9975
  Component 23: 0.9983
  Component 24: 0.9989
  Component 25: 0.9994
  Component 26: 0.9997
  Component 27: 0.9999
  Component 28: 1.0000
  Component 29: 1.0000
  Component 30: 1.0000

