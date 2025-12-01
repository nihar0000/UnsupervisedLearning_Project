import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

def run_supervised_learning(input_path):
    """
    Trains a CNN to classify faces.
    """
    print(f"Loading splits from {input_path}...")
    with open(input_path, 'rb') as f:
        splits = pickle.load(f)
        
    X_train = splits['X_train']
    y_train = splits['y_train']
    X_val = splits['X_val']
    y_val = splits['y_val']
    X_test = splits['X_test']
    y_test = splits['y_test']
    scaler = splits['scaler']
    
    # Reshape data for CNN (N, 112, 92, 1)
    # Note: Data was flattened (10304). We need to reshape it back.
    # Also, it was scaled. Reshaping scaled data is fine.
    
    img_height, img_width = 112, 92
    
    X_train_cnn = X_train.reshape(-1, img_height, img_width, 1)
    X_val_cnn = X_val.reshape(-1, img_height, img_width, 1)
    X_test_cnn = X_test.reshape(-1, img_height, img_width, 1)
    
    # One-hot encode labels
    num_classes = len(np.unique(y_train))
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    print(f"Input shape: {X_train_cnn.shape}")
    print(f"Number of classes: {num_classes}")
    
    # Build CNN
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    model.summary()
    
    # Train
    print("Training model...")
    history = model.fit(X_train_cnn, y_train_cat,
                        epochs=20,
                        batch_size=32,
                        validation_data=(X_val_cnn, y_val_cat))
                        
    # Plot training history
    plot_history(history)
    
    # Evaluate
    print("Evaluating on Test Set...")
    test_loss, test_acc = model.evaluate(X_test_cnn, y_test_cat)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Predictions
    y_pred_prob = model.predict(X_test_cnn)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('plots/confusion_matrix.png')
    print("Confusion matrix saved to plots/confusion_matrix.png")
    
    # Sample predictions
    show_sample_predictions(X_test_cnn, y_test, y_pred)

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training acc')
    plt.plot(epochs, val_acc, 'r*-', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'r*-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.savefig('plots/training_history.png')
    print("Training history saved to plots/training_history.png")

def show_sample_predictions(X, y_true, y_pred):
    indices = np.random.choice(len(X), 10, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        img = X[idx].reshape(112, 92)
        ax = axes[i]
        ax.imshow(img, cmap='gray')
        
        color = 'green' if y_true[idx] == y_pred[idx] else 'red'
        ax.set_title(f"True: {y_true[idx]}\nPred: {y_pred[idx]}", color=color)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig('plots/predictions.png')
    print("Sample predictions saved to plots/predictions.png")

if __name__ == "__main__":
    input_file = 'processed_data/splits.pkl'
    run_supervised_learning(input_file)
