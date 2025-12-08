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
    
    # --- IMPROVEMENT 1: L2 Normalization ---
    # This projects data onto a sphere, making cosine similarity equivalent to Euclidean distance
    from sklearn.preprocessing import Normalizer
    print("Applying L2 Normalization (Projecting to Unit Sphere)...")
    transformer = Normalizer().fit(X_train)
    X_train_norm = transformer.transform(X_train)
    
    # --- IMPROVEMENT 2: Use fewer dimensions ---
    # The last 50 components might be noise. Let's try top 30.
    X_train_clean = X_train_norm[:, :30] 
    print(f"Using top 30 Dimensions for Clustering (Shape: {X_train_clean.shape})")
    
    # We will cluster the training data to analyze how well it groups
    # Using the reduced data (PCA 95%)
    
    # --- K-Means (Improved) ---
    print("Running Improved K-Means...")
    # Determine optimal k using Elbow Method
    inertias = []
    k_range = range(10, 31) # We know there are 20 subjects, so check around 20
    
    for k in k_range:
        # print(f"Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_train_clean)
        inertias.append(kmeans.inertia_)
        
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k (Improved Data)')
    plt.grid(True)
    plt.savefig('plots/kmeans_elbow.png')
    print("K-Means elbow plot saved to plots/kmeans_elbow.png")
    
    # Let's choose k=20 (since we know the ground truth)
    k_optimal = 20
    kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_train_clean)
    
    # Calculate metrics
    sil_score = silhouette_score(X_train_clean, kmeans_labels)
    hom_score = homogeneity_score(y_train, kmeans_labels)
    print(f"K-Means (k={k_optimal}): Silhouette Score = {sil_score:.4f}, Homogeneity Score = {hom_score:.4f}")
    
    # Visualize K-Means clusters (using first 2 PCA components from the reduced data)
    plot_clusters(X_train[:, :2], kmeans_labels, 'K-Means Clustering (Normalized)', 'plots/kmeans_clusters.png')
    
    # Analyze cluster purity/composition
    analyze_clusters(kmeans_labels, y_train, "K-Means (Improved)")
    
    
    # --- DBSCAN (Improved) ---
    print("\nRunning Improved DBSCAN...")
    # Distance metric matters. With L2 norm, euclidean distance is related to cosine diff.
    
    # Heuristic: try a range of eps
    best_eps = 0.5
    best_sil = -1
    best_labels = None
    
    # Since we normalized, distances are small (0 to 1 range usually)
    # let's scan small EPS
    for eps in np.arange(0.1, 1.5, 0.05):
        dbscan = DBSCAN(eps=eps, min_samples=3)
        labels = dbscan.fit_predict(X_train_clean)
        
        if len(set(labels)) > 1:
             # Calculate silhouette excluding noise if needed, but standard is fine
             if len(set(labels)) == 1 and -1 in set(labels): continue
                 
             s = silhouette_score(X_train_clean, labels)
             n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
             # print(f"DBSCAN eps={eps:.2f} -> Clusters: {n_clusters}, Silhouette: {s:.4f}")
             
             if s > best_sil and n_clusters > 1:
                 best_sil = s
                 best_eps = eps
                 best_labels = labels
    
    if best_labels is not None:
        print(f"Best DBSCAN: eps={best_eps:.2f}, Silhouette={best_sil:.4f}")
        plot_clusters(X_train[:, :2], best_labels, f'DBSCAN Clustering (eps={best_eps:.2f})', 'plots/dbscan_clusters.png')
        analyze_clusters(best_labels, y_train, "DBSCAN (Improved)")
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
