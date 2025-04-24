import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('Dataset.csv')
df = df.loc[df["PRICE"].between(500000, 65000000)]

# Select relevant columns (excluding TYPE and SUBLOCALITY)
features = df[['PRICE', 'BEDS', 'BATH', 'MetriPatratiLocuinta']]

# Normalize the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# ===========================
# 1. Elbow Method for Optimal k
# ===========================
wcss = []  # List to store WCSS values

# Try different values of k from 1 to 10 (you can adjust this range)
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    wcss.append(kmeans.inertia_)  # Store the WCSS value for each k

# Plot the Elbow Curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k', fontsize=16)
plt.xlabel('Number of Clusters', fontsize=12)
plt.ylabel('WCSS (Inertia)', fontsize=12)
plt.grid(True)
plt.show()

# Based on the elbow plot, we choose an optimal k (e.g., 4 clusters)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(features_scaled)

# ===========================
# 2. Scatter Plot: PRICE vs MetriPatratiLocuinta (Colored by Cluster)
# ===========================
plt.figure(figsize=(10, 6))
plt.scatter(df['PRICE'], df['MetriPatratiLocuinta'], c=df['cluster'], cmap='viridis', s=50, alpha=0.6)
plt.title('Scatter Plot of PRICE vs MetriPatratiLocuinta (Colored by Cluster)', fontsize=16)
plt.xlabel('PRICE', fontsize=12)
plt.ylabel('MetriPatratiLocuinta (Square Meters)', fontsize=12)
plt.colorbar(label='Cluster')
plt.gcf().axes[0].xaxis.get_major_formatter().set_scientific(False)
plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
plt.grid(True)
plt.show()

# ===========================
# 3. Scatter Plot: PRICE vs BEDS (Colored by Cluster)
# ===========================
plt.figure(figsize=(10, 6))
plt.scatter(df['PRICE'], df['BEDS'], c=df['cluster'], cmap='viridis', s=50, alpha=0.6)
plt.title('Scatter Plot of PRICE vs BEDS (Colored by Cluster)', fontsize=16)
plt.xlabel('PRICE', fontsize=12)
plt.ylabel('BEDS', fontsize=12)
plt.colorbar(label='Cluster')
plt.gcf().axes[0].xaxis.get_major_formatter().set_scientific(False)
plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
plt.grid(True)
plt.show()

# ===========================
# 4. Scatter Plot: MetriPatratiLocuinta vs BEDS (Colored by Cluster)
# ===========================
plt.figure(figsize=(10, 6))
plt.scatter(df['MetriPatratiLocuinta'], df['BEDS'], c=df['cluster'], cmap='viridis', s=50, alpha=0.6)
plt.title('Scatter Plot of MetriPatratiLocuinta vs BEDS (Colored by Cluster)', fontsize=16)
plt.xlabel('MetriPatratiLocuinta (Square Meters)', fontsize=12)
plt.ylabel('BEDS', fontsize=12)
plt.colorbar(label='Cluster')
plt.gcf().axes[0].xaxis.get_major_formatter().set_scientific(False)
plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
plt.grid(True)
plt.show()

# ===========================
# 5. PCA for Dimensionality Reduction (Optional: If you want a 2D projection)
# ===========================
pca = PCA(n_components=2)
pca_components = pca.fit_transform(features_scaled)
pca_df = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])
pca_df['cluster'] = df['cluster']

# Plot the PCA results
plt.figure(figsize=(10, 6))
plt.scatter(pca_df['PCA1'], pca_df['PCA2'], c=pca_df['cluster'], cmap='viridis', s=50, alpha=0.6)
plt.title('PCA Projection of Housing Data (Colored by Cluster)', fontsize=16)
plt.xlabel('PCA1', fontsize=12)
plt.ylabel('PCA2', fontsize=12)
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()