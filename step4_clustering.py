import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, homogeneity_score
import os

def run_clustering(input_path):
    """
    Applies K-Means and DBSCAN clustering on the reduced dataset.
    """
    print(f"Loading reduced data from {input_path}...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
        
    X_train = data['X_train']
    y_train = data['y_train']
    
    # We will cluster the training data to analyze how well it groups
    # Using the reduced data (PCA 95%)
    
    # --- K-Means ---
    print("Running K-Means...")
    # Determine optimal k using Elbow Method
    inertias = []
    k_range = range(10, 31) # We know there are 20 subjects, so check around 20
    
    for k in k_range:
        print(f"Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_train)
        inertias.append(kmeans.inertia_)
        
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.savefig('plots/kmeans_elbow.png')
    print("K-Means elbow plot saved to plots/kmeans_elbow.png")
    
    # Let's choose k=20 (since we know the ground truth) for detailed analysis
    k_optimal = 20
    kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_train)
    
    # Calculate metrics
    sil_score = silhouette_score(X_train, kmeans_labels)
    hom_score = homogeneity_score(y_train, kmeans_labels)
    print(f"K-Means (k={k_optimal}): Silhouette Score = {sil_score:.4f}, Homogeneity Score = {hom_score:.4f}")
    
    # Visualize K-Means clusters (using first 2 PCA components from the reduced data)
    plot_clusters(X_train[:, :2], kmeans_labels, 'K-Means Clustering (k=20)', 'plots/kmeans_clusters.png')
    
    # Analyze cluster purity/composition
    analyze_clusters(kmeans_labels, y_train, "K-Means")
    
    # --- DBSCAN ---
    print("Running DBSCAN...")
    # DBSCAN is sensitive to eps. We can use k-distance graph to find eps, but let's try a few
    # Since data is high dimensional (even after PCA ~100 dims), distance metric matters.
    # We might want to use cosine distance or just euclidean.
    
    # Heuristic: try a range of eps
    best_eps = 0.5
    best_sil = -1
    best_labels = None
    
    # Distances in high dim can be large. 
    # Let's estimate distances.
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=5)
    nbrs = neigh.fit(X_train)
    distances, indices = nbrs.kneighbors(X_train)
    distances = np.sort(distances[:, 4], axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title('K-distance Graph for DBSCAN')
    plt.xlabel('Points sorted by distance')
    plt.ylabel('5th Nearest Neighbor Distance')
    plt.grid(True)
    plt.savefig('plots/dbscan_kdist.png')
    print("DBSCAN k-distance plot saved to plots/dbscan_kdist.png")
    
    # From the plot (visual inspection needed, but let's guess based on typical values for normalized data)
    # If data is standardized, distances are roughly sqrt(d). 
    # Let's try eps values around the "knee" of the curve. 
    # We'll sweep a bit.
    
    for eps in np.arange(5, 25, 1): # Adjust range based on data scale
        dbscan = DBSCAN(eps=eps, min_samples=3)
        labels = dbscan.fit_predict(X_train)
        
        # Ignore noise (-1) for silhouette if possible, or include it
        if len(set(labels)) > 1: # at least 2 clusters (including noise)
             # Filter out noise for score calculation if it dominates, but here we just calculate
             # If all noise, skip
             if len(set(labels)) == 1 and -1 in set(labels):
                 continue
                 
             s = silhouette_score(X_train, labels)
             n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
             print(f"DBSCAN eps={eps}, min_samples=3 -> Clusters: {n_clusters}, Silhouette: {s:.4f}")
             
             if s > best_sil and n_clusters > 1:
                 best_sil = s
                 best_eps = eps
                 best_labels = labels
    
    if best_labels is not None:
        print(f"Best DBSCAN: eps={best_eps}, Silhouette={best_sil:.4f}")
        plot_clusters(X_train[:, :2], best_labels, f'DBSCAN Clustering (eps={best_eps})', 'plots/dbscan_clusters.png')
        analyze_clusters(best_labels, y_train, "DBSCAN")
    else:
        print("DBSCAN failed to find meaningful clusters.")

def plot_clusters(X_2d, labels, title, save_path):
    plt.figure(figsize=(10, 8))
    # Handle noise (-1) separately if needed, but scatter handles it
    unique_labels = np.unique(labels)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        my_members = labels == k
        if k == -1:
            col = [0, 0, 0, 1] # Black for noise
            label_name = "Noise"
        else:
            label_name = f"Cluster {k}"
            
        plt.scatter(X_2d[my_members, 0], X_2d[my_members, 1], c=[col], label=label_name, s=30, alpha=0.6)
        
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    # plt.legend() # Legend might be too big
    plt.savefig(save_path)
    print(f"Cluster plot saved to {save_path}")

def analyze_clusters(cluster_labels, true_labels, method_name):
    """
    Analyzes the purity/composition of clusters.
    """
    print(f"\n--- {method_name} Analysis ---")
    df = pd.DataFrame({'Cluster': cluster_labels, 'True Label': true_labels})
    
    # Purity: For each cluster, find most frequent class
    total_samples = len(true_labels)
    correct_samples = 0
    
    for cluster in np.unique(cluster_labels):
        if cluster == -1: continue # Skip noise
        
        cluster_data = df[df['Cluster'] == cluster]
        most_frequent_label = cluster_data['True Label'].mode()[0]
        count = cluster_data[cluster_data['True Label'] == most_frequent_label].shape[0]
        
        correct_samples += count
        purity = count / len(cluster_data)
        
        print(f"Cluster {cluster}: Most frequent Subject {most_frequent_label} ({count}/{len(cluster_data)} = {purity:.2f})")
        
    overall_purity = correct_samples / total_samples # Note: noise counts as incorrect here
    print(f"Overall Purity: {overall_purity:.4f}")

if __name__ == "__main__":
    input_file = 'processed_data/reduced_data.pkl'
    run_clustering(input_file)
