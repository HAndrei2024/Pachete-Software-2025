import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

def preprocess_kmeans(df):
    df_kmeans = df.copy()
    df_kmeans = df_kmeans.loc[df_kmeans["PRICE"].between(500000, 65000000)]
    features = df_kmeans[['PRICE', 'BEDS', 'BATH', 'MetriPatratiLocuinta']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return df_kmeans, features_scaled

def plot_elbow(features_scaled):
    wcss = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(features_scaled)
        wcss.append(km.inertia_)

    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss, marker='o')
    ax.set_title('Elbow Method for Optimal k')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('WCSS (Inertia)')
    ax.grid(True)
    return fig

def cluster_and_add(df_kmeans, features_scaled, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_kmeans['cluster'] = kmeans.fit_predict(features_scaled)
    return df_kmeans

def plot_price_vs_metripatrat(df_kmeans):
    fig, ax = plt.subplots()
    scatter = ax.scatter(df_kmeans['PRICE'], df_kmeans['MetriPatratiLocuinta'],
                         c=df_kmeans['cluster'], cmap='viridis', s=50, alpha=0.6)
    ax.set_title('PRICE vs MetriPatratiLocuinta')
    ax.set_xlabel('PRICE')
    ax.set_ylabel('MetriPatratiLocuinta')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    ax.grid(True)
    return fig

def plot_price_vs_beds(df_kmeans):
    fig, ax = plt.subplots()
    scatter = ax.scatter(df_kmeans['PRICE'], df_kmeans['BEDS'],
                         c=df_kmeans['cluster'], cmap='viridis', s=50, alpha=0.6)
    ax.set_title('PRICE vs BEDS')
    ax.set_xlabel('PRICE')
    ax.set_ylabel('BEDS')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    ax.grid(True)
    return fig

def plot_metripatrat_vs_beds(df_kmeans):
    fig, ax = plt.subplots()
    scatter = ax.scatter(df_kmeans['MetriPatratiLocuinta'], df_kmeans['BEDS'],
                         c=df_kmeans['cluster'], cmap='viridis', s=50, alpha=0.6)
    ax.set_title('MetriPatratiLocuinta vs BEDS')
    ax.set_xlabel('MetriPatratiLocuinta')
    ax.set_ylabel('BEDS')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    ax.grid(True)
    return fig

def plot_pca_projection(features_scaled, df_kmeans):
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(features_scaled)
    pca_df = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])
    pca_df['cluster'] = df_kmeans['cluster']

    fig, ax = plt.subplots()
    scatter = ax.scatter(pca_df['PCA1'], pca_df['PCA2'],
                         c=pca_df['cluster'], cmap='viridis', s=50, alpha=0.6)
    ax.set_title('PCA Projection of Housing Data')
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    ax.grid(True)
    return fig
