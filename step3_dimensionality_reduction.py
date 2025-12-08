import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import os

def run_dimensionality_reduction(input_path, output_path):
    """
    Applies PCA and Autoencoder for dimensionality reduction.
    """
    print(f"Loading splits from {input_path}...")
    with open(input_path, 'rb') as f:
        splits = pickle.load(f)
        
    X_train = splits['X_train']
    y_train = splits['y_train']
    # We will use the full dataset for visualization later, but fit on train
    X_val = splits['X_val']
    y_val = splits['y_val']
    X_test = splits['X_test']
    y_test = splits['y_test']
    
    # --- PCA ---
    print("Running PCA...")
    # Fit PCA with enough components to see variance
    pca = PCA(n_components=100)
    pca.fit(X_train)
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.savefig('plots/pca_variance.png')
    print("PCA variance plot saved to plots/pca_variance.png")
    
    # Transform data using top 2 components for visualization
    pca_2d = PCA(n_components=2)
    X_train_pca = pca_2d.fit_transform(X_train)
    
    plot_2d_projection(X_train_pca, y_train, 'PCA Projection (Train Set)', 'plots/pca_projection.png')
    
    # --- Autoencoder ---
    print("Running Autoencoder...")
    input_dim = X_train.shape[1]
    encoding_dim = 64 # Compressed representation size
    
    # Simple Autoencoder
    input_img = Input(shape=(input_dim,))
    encoded = Dense(256, activation='relu')(input_img)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded) # Latent space
    
    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(256, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='linear')(decoded) # Linear for reconstruction of scaled data
    
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    history = autoencoder.fit(X_train, X_train,
                    epochs=50,
                    batch_size=32,
                    shuffle=True,
                    validation_data=(X_val, X_val),
                    verbose=0)
                    
    print("Autoencoder training complete.")
    
    # Encode data
    X_train_ae = encoder.predict(X_train)
    
    # For visualization, we might need further reduction if latent dim > 2
    # Let's use PCA on the AE latent space to visualize in 2D
    pca_ae = PCA(n_components=2)
    X_train_ae_2d = pca_ae.fit_transform(X_train_ae)
    
    plot_2d_projection(X_train_ae_2d, y_train, 'Autoencoder Latent Projection (PCA reduced)', 'plots/ae_projection.png')
    
    # Save reduced data (using PCA 95% variance or similar for clustering)
    # For clustering, let's use a reasonable number of components, e.g., 50 or explained variance > 0.95
    pca_final = PCA(n_components=0.95)
    X_train_reduced = pca_final.fit_transform(X_train)
    X_val_reduced = pca_final.transform(X_val)
    X_test_reduced = pca_final.transform(X_test)
    
    print(f"Reduced dimensions from {input_dim} to {X_train_reduced.shape[1]} (95% variance)")
    
    reduced_data = {
        'X_train': X_train_reduced, 'y_train': y_train,
        'X_val': X_val_reduced, 'y_val': y_val,
        'X_test': X_test_reduced, 'y_test': y_test,
        'pca_model': pca_final,
        'ae_encoder': encoder # Saving encoder just in case
    }
    
    print(f"Saving reduced data to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(reduced_data, f)
    print("Done.")

def plot_2d_projection(X, y, title, save_path):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='jet', alpha=0.7)
    plt.colorbar(scatter, label='Subject ID')
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig(save_path)
    print(f"Projection plot saved to {save_path}")

if __name__ == "__main__":
    input_file = 'processed_data/splits.pkl'
    output_file = 'processed_data/reduced_data.pkl'
    
    run_dimensionality_reduction(input_file, output_file)
