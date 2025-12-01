import scipy.io
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def load_and_prepare_data(mat_file_path, output_path):
    """
    Loads the UMIST dataset, converts it to a DataFrame, and saves it.
    """
    print(f"Loading dataset from {mat_file_path}...")
    mat = scipy.io.loadmat(mat_file_path)
    facedat = mat['facedat'][0]  # Access the cell array
    
    data = []
    labels = []
    
    print(f"Found {len(facedat)} subjects.")
    
    for subject_id, subject_images in enumerate(facedat):
        # subject_images is (112, 92, N)
        num_images = subject_images.shape[2]
        print(f"Subject {subject_id}: {num_images} images")
        
        for i in range(num_images):
            img = subject_images[:, :, i]
            # Flatten the image
            img_flat = img.flatten()
            data.append(img_flat)
            labels.append(subject_id)
            
    data = np.array(data)
    labels = np.array(labels)
    
    print(f"Total images: {data.shape[0]}")
    print(f"Image dimensions (flattened): {data.shape[1]}")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df['label'] = labels
    
    # Save to pickle
    print(f"Saving processed dataset to {output_path}...")
    df.to_pickle(output_path)
    print("Done.")

    # Show a few sample images
    show_sample_images(data, labels)

def show_sample_images(data, labels):
    """
    Displays a few sample images from the dataset.
    """
    unique_labels = np.unique(labels)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(unique_labels):
            label = unique_labels[i]
            # Get first image for this label
            img_flat = data[labels == label][0]
            img = img_flat.reshape(112, 92)
            
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Subject {label}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('plots/sample_images.png')
    print("Sample images saved to plots/sample_images.png")
    # plt.show() # Commented out for non-interactive execution

if __name__ == "__main__":
    if not os.path.exists('plots'):
        os.makedirs('plots')
        
    mat_file = 'umist_cropped.mat'
    output_file = 'processed_data/dataset.pkl'
    
    load_and_prepare_data(mat_file, output_file)
