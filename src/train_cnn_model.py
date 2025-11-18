import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Install TensorFlow/Keras if not present
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
    from tensorflow.keras.optimizers import Adam
except ImportError:
    print("TensorFlow/Keras not found. Please install with: pip install tensorflow")
    exit()

# --- Configuration ---
DATA_PATH = 'data/er_triage_dataset_15k_amman.csv'
MODEL_DIR = 'models'
REPORT_PATH = 'docs/cnn_model_evaluation_report_thesis.txt'
CM_PATH = 'docs/cnn_confusion_matrix_thesis.png'
EPOCHS = 10
BATCH_SIZE = 32

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(path):
    """Loads the dataset and prepares it for the CNN."""
    try:
        df = pd.read_csv(path)
        print(f"Data loaded successfully from {path}. Shape: {df.shape}")
        
        # Define features (X) and target (y)
        features = [col for col in df.columns if col not in ['Risk_Label']]
        X = df[features].values
        y = df['Risk_Label'].values
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Reshape for 1D CNN: (samples, timesteps, features)
        # Here, 'timesteps' is 1 and 'features' is the number of vital signs
        X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        return X_cnn, y, features
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}")
        return None, None, None

def build_cnn_model(input_shape):
    """Builds a simple 1D Convolutional Neural Network model."""
    model = Sequential([
        # First Conv layer
        Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=input_shape), # Reduced filters
        Dropout(0.5), # Increased dropout
        
        # Second Conv layer
        Conv1D(filters=32, kernel_size=3, activation='relu'), # Reduced filters
        Dropout(0.6), # Increased dropout
        
        # Flatten and Dense layers
        Flatten(),
        Dense(32, activation='relu'), # Reduced dense layer size
        Dropout(0.7), # Increased dropout
        Dense(1, activation='sigmoid') # Binary classification output
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_evaluate_model(X, y, features):
    """Trains and evaluates the CNN model."""
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # Build and train model
    model = build_cnn_model(input_shape=(X_train.shape[1], 1))
    model.summary()
    
    print("Starting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1, # Use 10% of training data for validation
        verbose=1
    )
    print("Model training complete.")
    
    # --- Model Evaluation ---
    y_proba = model.predict(X_test).flatten()
    y_pred = (y_proba > 0.5).astype(int)
    
    # 1. Classification Report
    report_text = classification_report(y_test, y_pred, target_names=['Low Risk (0)', 'High Risk (1)'])
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # 3. AUC-ROC Score
    auc_roc = roc_auc_score(y_test, y_proba)
    
    # --- Save Results ---
    
    # Save text report
    with open(REPORT_PATH, 'w') as f:
        f.write("--- AI ER Triage CNN Model Evaluation Report ---\n\n")
        f.write(f"Model: 1D Convolutional Neural Network\n")
        f.write(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}\n")
        f.write(f"AUC-ROC Score: {auc_roc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_text)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
        
    print(f"Evaluation report saved to {REPORT_PATH}")
    
    # Plot and save Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Low Risk', 'Predicted High Risk'], 
                yticklabels=['Actual Low Risk', 'Actual High Risk'])
    plt.title('Confusion Matrix for CNN ER Triage Model')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(CM_PATH)
    plt.close()
    print(f"Confusion Matrix plot saved to {CM_PATH}")
    
    return model

if __name__ == '__main__':
    # 1. Load Data
    X_cnn, y, features = load_data(DATA_PATH)
    
    if X_cnn is not None:
        # 2. Train and Evaluate Model
        trained_model = train_and_evaluate_model(X_cnn, y, features)
        
        print("\nCNN model development and evaluation complete.")
