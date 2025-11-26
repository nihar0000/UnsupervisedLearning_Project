"""
Step 1: Data Preparation [2 points]
Load the umist_cropped.mat dataset and explore its structure.
Convert images and labels into a Pandas DataFrame.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

def load_umist_dataset(file_path='umist_cropped.mat'):
    """
    Load the UMIST face dataset from .mat file
    
    Parameters:
    -----------
    file_path : str
        Path to the umist_cropped.mat file
        
    Returns:
    --------
    data : dict
        Dictionary containing the loaded data
    """
    print("Loading UMIST dataset...")
    data = loadmat(file_path)
    print(f"Dataset keys: {data.keys()}")
    return data

def explore_dataset_structure(data):
    """
    Explore and print the structure of the dataset
    
    Parameters:
    -----------
    data : dict
        Dictionary containing the loaded data
    """
    print("\n=== Dataset Structure ===")
    for key in data.keys():
        if not key.startswith('__'):
            print(f"\nKey: {key}")
            print(f"Type: {type(data[key])}")
            if isinstance(data[key], np.ndarray):
                print(f"Shape: {data[key].shape}")
                print(f"Data type: {data[key].dtype}")

def create_dataframe(data):
    """
    Convert images and labels into a Pandas DataFrame
    
    Parameters:
    -----------
    data : dict
        Dictionary containing the loaded data
        
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with image features and labels
    images : np.ndarray
        Original images array
    labels : np.ndarray
        Labels array
    """
    # Extract images and labels (adjust keys based on actual dataset structure)
    # UMIST cropped dataset structure: 'facedat' contains arrays of images per person
    if 'facedat' in data:
        # facedat is structured as array of arrays, one per person
        facedat = data['facedat'][0]  # Remove outer dimension
        dirnames = data['dirnames'][0] if 'dirnames' in data else None
        
        # First pass: collect all images and find consistent dimensions
        all_images = []
        all_labels = []
        image_shapes = []
        
        for person_idx, person_images in enumerate(facedat):
            # Each person_images is an array of images for that person
            # Handle different array structures
            if isinstance(person_images, np.ndarray):
                if len(person_images.shape) == 3:
                    # Shape: (n_images, height, width)
                    n_images = person_images.shape[0]
                    for img_idx in range(n_images):
                        img = person_images[img_idx]
                        all_images.append(img.flatten())  # Flatten immediately
                        all_labels.append(person_idx)
                        image_shapes.append(img.shape)
                elif len(person_images.shape) == 2:
                    # Single image (height, width)
                    all_images.append(person_images.flatten())  # Flatten immediately
                    all_labels.append(person_idx)
                    image_shapes.append(person_images.shape)
        
        # Convert to arrays (now all images are flattened)
        # Find the most common image shape
        from collections import Counter
        shape_counts = Counter([s[0]*s[1] for s in image_shapes])
        target_size = shape_counts.most_common(1)[0][0]
        
        # Filter to keep only images with the target size
        filtered_images = []
        filtered_labels = []
        original_shape = None
        
        for img, label, shape in zip(all_images, all_labels, image_shapes):
            if len(img) == target_size:
                filtered_images.append(img)
                filtered_labels.append(label)
                if original_shape is None:
                    original_shape = shape
        
        images = np.array(filtered_images)
        labels = np.array(filtered_labels)
        
        # Reshape for visualization later (keep track of original dimensions)
        if original_shape is not None:
            print(f"Using images of size: {original_shape}")
            print(f"Filtered dataset: {len(filtered_images)} images (removed {len(all_images) - len(filtered_images)} inconsistent images)")
        
    elif 'fea' in data:
        images = data['fea']
        labels = data['gnd'].flatten() if 'gnd' in data else None
    elif 'X' in data:
        images = data['X']
        labels = data['y'].flatten() if 'y' in data else None
    else:
        # Try to find the main data arrays
        non_meta_keys = [k for k in data.keys() if not k.startswith('__')]
        print(f"Available keys: {non_meta_keys}")
        images = data[non_meta_keys[0]]
        if len(non_meta_keys) > 1:
            labels = data[non_meta_keys[1]].flatten()
        else:
            labels = None
    
    print(f"\n=== Data Summary ===")
    print(f"Images shape: {images.shape}")
    print(f"Number of samples: {images.shape[0]}")
    if len(images.shape) > 1:
        print(f"Feature dimensions: {images.shape[1]}")
    
    if labels is not None:
        print(f"Labels shape: {labels.shape}")
        print(f"Number of unique persons: {len(np.unique(labels))}")
        print(f"Label range: {labels.min()} to {labels.max()}")
        print(f"Samples per person:")
        unique, counts = np.unique(labels, return_counts=True)
        for person, count in zip(unique, counts):
            print(f"  Person {person}: {count} images")
    
    # Images are already flattened in the new structure
    flattened_images = images
    
    # Create column names for pixel features
    n_features = flattened_images.shape[1]
    feature_columns = [f'pixel_{i}' for i in range(n_features)]
    
    # Create DataFrame
    df = pd.DataFrame(flattened_images, columns=feature_columns)
    
    if labels is not None:
        df['label'] = labels
    
    print(f"\n=== DataFrame Info ===")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {len(df.columns)} ({len(df.columns)-1} features + 1 label)")
    
    return df, images, labels

def visualize_sample_images(images, labels, n_samples=10):
    """
    Visualize sample images from the dataset
    
    Parameters:
    -----------
    images : np.ndarray
        Array of images
    labels : np.ndarray
        Array of labels
    n_samples : int
        Number of samples to display
    """
    # Images should already be in correct 3D shape
    images_reshaped = images
    
    n_samples = min(n_samples, len(images))
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Sample Images from UMIST Dataset', fontsize=16)
    
    # Select random samples
    indices = np.random.choice(len(images), n_samples, replace=False)
    
    for idx, ax in enumerate(axes.flat):
        if idx < n_samples:
            img_idx = indices[idx]
            ax.imshow(images_reshaped[img_idx], cmap='gray')
            if labels is not None:
                ax.set_title(f'Person {labels[img_idx]}')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('step1_sample_images.png', dpi=300, bbox_inches='tight')
    print("\n✓ Sample images saved as 'step1_sample_images.png'")
    plt.show()

def visualize_person_samples(images, labels, person_id, n_samples=5):
    """
    Visualize multiple images of a specific person
    
    Parameters:
    -----------
    images : np.ndarray
        Array of images
    labels : np.ndarray
        Array of labels
    person_id : int
        ID of the person to visualize
    n_samples : int
        Number of samples to display
    """
    # Images should already be in correct 3D shape
    images_reshaped = images
    
    # Get all images of the specified person
    person_indices = np.where(labels == person_id)[0]
    n_samples = min(n_samples, len(person_indices))
    
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 3))
    fig.suptitle(f'Sample Images of Person {person_id}', fontsize=14)
    
    for idx in range(n_samples):
        img_idx = person_indices[idx]
        if n_samples == 1:
            ax = axes
        else:
            ax = axes[idx]
        ax.imshow(images_reshaped[img_idx], cmap='gray')
        ax.set_title(f'Image {idx+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'step1_person_{person_id}_samples.png', dpi=300, bbox_inches='tight')
    print(f"✓ Person {person_id} samples saved as 'step1_person_{person_id}_samples.png'")
    plt.show()

def main():
    """
    Main function to execute data preparation steps
    """
    print("=" * 60)
    print("STEP 1: DATA PREPARATION")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists('umist_cropped.mat'):
        print("ERROR: umist_cropped.mat not found in the current directory!")
        print("Please download the dataset and place it in the project folder.")
        return
    
    # Load dataset
    data = load_umist_dataset()
    
    # Explore structure
    explore_dataset_structure(data)
    
    # Create DataFrame
    df, images, labels = create_dataframe(data)
    
    # Save DataFrame to CSV
    df.to_csv('umist_dataframe.csv', index=False)
    print(f"\n✓ DataFrame saved as 'umist_dataframe.csv'")
    
    # Visualize sample images (images is already flattened in DataFrame)
    # For visualization, we need to reshape back - infer from create_dataframe
    n_pixels = df.shape[1] - 1  # Exclude label column
    if n_pixels == 10304:  # 112 x 92
        img_height, img_width = 112, 92
    elif n_pixels == 2392:  # 92 x 26
        img_height, img_width = 92, 26
    else:
        # Try square-ish
        img_height = int(np.sqrt(n_pixels))
        img_width = n_pixels // img_height
    
    # Reshape images for visualization
    images_for_viz = df.drop('label', axis=1).values.reshape(-1, img_height, img_width)
    labels_for_viz = df['label'].values
    
    visualize_sample_images(images_for_viz, labels_for_viz, n_samples=10)
    
    # Visualize images of a specific person
    if labels is not None:
        unique_labels = np.unique(labels_for_viz)
        if len(unique_labels) > 0:
            sample_person = unique_labels[0]
            visualize_person_samples(images_for_viz, labels_for_viz, sample_person, n_samples=5)
    
    print("\n" + "=" * 60)
    print("✓ STEP 1 COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nOutputs generated:")
    print("  - umist_dataframe.csv")
    print("  - step1_sample_images.png")
    print("  - step1_person_X_samples.png")
    print("\nNext: Run step2_data_splitting.py")

if __name__ == "__main__":
    main()
