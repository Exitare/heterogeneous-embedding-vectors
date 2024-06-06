import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from pathlib import Path
from sklearn.decomposition import PCA


if __name__ == '__main__':
    brca_cancer = pd.read_csv(Path("results", "embeddings", "cancer", "brca_embeddings.csv"))
    blca_cancer = pd.read_csv(Path("results", "embeddings","cancer", "THCA_embeddings.csv"))

    # Combine datasets
    data = pd.concat([brca_cancer, blca_cancer])
    labels = np.array([0] * len(brca_cancer) + [1] * len(blca_cancer))
    # Calculate cosine distances
    cosine_dist = cosine_distances(data)

    # Perform KMeans clustering
    num_clusters = 2  # Set the number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(data)
    labels = kmeans.labels_

    # Calculate intra-cluster and inter-cluster distances
    intra_cluster_distances = []
    inter_cluster_distances = []

    for i in range(num_clusters):
        cluster_points = data[labels == i]
        other_points = data[labels != i]
        intra_dist = cosine_distances(cluster_points)
        intra_cluster_distances.extend(intra_dist[np.triu_indices(len(cluster_points), k=1)])
        if other_points.size > 0:
            inter_dist = cosine_distances(cluster_points, other_points)
            inter_cluster_distances.extend(inter_dist.flatten())

    # remove datapoints with distance greater than 3std
    intra_cluster_distances = np.array(intra_cluster_distances)
    inter_cluster_distances = np.array(inter_cluster_distances)

    intra_cluster_distances = intra_cluster_distances[intra_cluster_distances < np.mean(intra_cluster_distances) + 3 * np.std(intra_cluster_distances)]
    inter_cluster_distances = inter_cluster_distances[inter_cluster_distances < np.mean(inter_cluster_distances) + 3 * np.std(inter_cluster_distances)]

    # Plotting the distances
    plt.figure(figsize=(12, 6))

    sns.histplot(intra_cluster_distances, kde=True, color='blue', label='Intra-cluster Distances')
    sns.histplot(inter_cluster_distances, kde=True, color='red', label='Inter-cluster Distances')

    plt.title('Cosine Distances within and between Clusters')
    plt.xlabel('Cosine Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # Output the average distances
    print(f"Average Intra-cluster Distance: {np.mean(intra_cluster_distances)}")
    print(f"Average Inter-cluster Distance: {np.mean(inter_cluster_distances)}")

    # plot the kmeans cluster
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(data)
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels)
    plt.title('PCA of Embeddings with KMeans Labels')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

