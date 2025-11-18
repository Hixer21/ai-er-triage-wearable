import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- Configuration ---
DATA_PATH = 'data/er_triage_dataset_15k.csv'
MODEL_DIR = 'models'
REPORT_PATH = 'docs/model_evaluation_report.txt'
CM_PATH = 'docs/confusion_matrix.png'

# Create model directory if it doesn't exist
# os.makedirs(MODEL_DIR, exist_ok=True) # Cannot create directories due to permission issues. We will rely on the existing 'models' directory.

def load_data(path):
    """Loads the dataset."""
    try:
        df = pd.read_csv(path)
        print(f"Data loaded successfully from {path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}")
        return None

def train_and_evaluate_model(df):
    """Trains a Random Forest Classifier and evaluates its performance."""
    
    # Define features (X) and target (y)
    features = [col for col in df.columns if col not in ['Risk_Label']]
    X = df[features]
    y = df['Risk_Label']
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # --- Model Training: Random Forest Classifier ---
    # Random Forest is a robust choice for initial classification tasks
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced' # Important for imbalanced data (80/20 split)
    )
    
    print("Starting model training...")
    model.fit(X_train, y_train)
    print("Model training complete.")
    
    # --- Model Evaluation ---
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 1. Classification Report
    report = classification_report(y_test, y_pred, target_names=['Low Risk (0)', 'High Risk (1)'], output_dict=True)
    report_text = classification_report(y_test, y_pred, target_names=['Low Risk (0)', 'High Risk (1)'])
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # 3. AUC-ROC Score
    auc_roc = roc_auc_score(y_test, y_proba)
    
    # --- Save Results ---
    
    # Save text report
    with open(REPORT_PATH, 'w') as f:
        f.write("--- AI ER Triage Model Evaluation Report ---\n\n")
        f.write(f"Model: Random Forest Classifier\n")
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
    plt.title('Confusion Matrix for ER Triage Model')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(CM_PATH)
    plt.close()
    print(f"Confusion Matrix plot saved to {CM_PATH}")
    
    # Feature Importance (optional but useful)
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    print("\nTop 5 Feature Importances:")
    print(feature_importance_df.head())
    
    # Save feature importance to report
    with open(REPORT_PATH, 'a') as f:
        f.write("\n\nFeature Importances:\n")
        f.write(feature_importance_df.to_string(index=False))
        
    print("Feature importances appended to report.")
    
    return model

if __name__ == '__main__':
    # 1. Load Data
    data = load_data(DATA_PATH)
    
    if data is not None:
        # 2. Train and Evaluate Model
        trained_model = train_and_evaluate_model(data)
        
        # 3. Save Model (Placeholder - not strictly necessary for this task but good practice)
        # import joblib
        # joblib.dump(trained_model, os.path.join(MODEL_DIR, 'random_forest_triage_model.joblib'))
        # print(f"Trained model saved to {MODEL_DIR}/random_forest_triage_model.joblib")
        
        print("\nAI model development and evaluation complete.")
