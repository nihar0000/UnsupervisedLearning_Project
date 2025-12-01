import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import os

def split_and_normalize(input_path, output_path):
    """
    Splits the dataset into Train, Validation, and Test sets using stratified sampling.
    Normalizes the data using StandardScaler.
    """
    print(f"Loading dataset from {input_path}...")
    df = pd.read_pickle(input_path)
    
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    print(f"Data shape: {X.shape}, Labels shape: {y.shape}")
    
    # First split: Train (70%) and Temp (30%)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_index, temp_index = next(sss1.split(X, y))
    
    X_train, X_temp = X[train_index], X[temp_index]
    y_train, y_temp = y[train_index], y[temp_index]
    
    # Second split: Validation (15%) and Test (15%) - from the 30% temp
    # 0.5 of 30% is 15% of total
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_index, test_index = next(sss2.split(X_temp, y_temp))
    
    X_val, X_test = X_temp[val_index], X_temp[test_index]
    y_val, y_test = y_temp[val_index], y_temp[test_index]
    
    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Normalize
    print("Normalizing data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save splits
    splits = {
        'X_train': X_train_scaled, 'y_train': y_train,
        'X_val': X_val_scaled, 'y_val': y_val,
        'X_test': X_test_scaled, 'y_test': y_test,
        'scaler': scaler
    }
    
    print(f"Saving splits to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(splits, f)
        
    # Visualize distributions
    plot_distributions(y_train, y_val, y_test)
    print("Done.")

def plot_distributions(y_train, y_val, y_test):
    """
    Plots the distribution of images per person in each subset.
    """
    unique_labels = np.unique(np.concatenate([y_train, y_val, y_test]))
    
    train_counts = [np.sum(y_train == l) for l in unique_labels]
    val_counts = [np.sum(y_val == l) for l in unique_labels]
    test_counts = [np.sum(y_test == l) for l in unique_labels]
    
    x = np.arange(len(unique_labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, train_counts, width, label='Train')
    rects2 = ax.bar(x, val_counts, width, label='Validation')
    rects3 = ax.bar(x + width, test_counts, width, label='Test')
    
    ax.set_ylabel('Number of Images')
    ax.set_xlabel('Subject ID')
    ax.set_title('Distribution of Images per Subject in Splits')
    ax.set_xticks(x)
    ax.set_xticklabels(unique_labels)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('plots/split_distribution.png')
    print("Distribution plot saved to plots/split_distribution.png")

if __name__ == "__main__":
    input_file = 'processed_data/dataset.pkl'
    output_file = 'processed_data/splits.pkl'
    
    split_and_normalize(input_file, output_file)
