import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import xgboost as xgb
from urllib.request import urlretrieve
import warnings
warnings.filterwarnings('ignore')

# Download and load the Wine Quality Dataset
def load_dataset():
    """Load the Wine Quality Dataset from UCI Repository"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    if not os.path.exists('winequality-red.csv'):
        print("Downloading dataset...")
        urlretrieve(url, 'winequality-red.csv')
    
    df = pd.read_csv('winequality-red.csv', sep=';')
    return df

# Preprocess the data
def preprocess_data(df):
    """Preprocess the wine quality dataset"""
    # Create binary classification (Good/Bad wine)
    # Quality > 6 = Good (1), Quality <= 6 = Bad (0)
    df['quality_binary'] = (df['quality'] > 6).astype(int)
    
    # Separate features and target
    X = df.drop(['quality', 'quality_binary'], axis=1)
    y = df['quality_binary']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Train and evaluate models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train all 6 models and evaluate them"""
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbor': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # AUC score (only if predict_proba is available)
        if y_pred_proba is not None:
            auc = roc_auc_score(y_test, y_pred_proba)
        else:
            auc = 0.0
        
        results[name] = {
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mcc': mcc,
            'model': model
        }
        
        print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    
    return results, trained_models

# Save models and results
def save_models_and_results(results, scaler):
    """Save trained models and results to pickle files"""
    
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Save individual models
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'K-Nearest Neighbor': 'k-nearest_neighbor.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl'
    }
    
    for name, filename in model_files.items():
        with open(f'model/{filename}', 'wb') as f:
            pickle.dump(results[name]['model'], f)
    
    # Save scaler
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save results (without model objects to reduce file size)
    results_for_saving = {}
    for name, result in results.items():
        results_for_saving[name] = {
            'accuracy': result['accuracy'],
            'auc': result['auc'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1': result['f1'],
            'mcc': result['mcc']
        }
    
    with open('model/model_results.pkl', 'wb') as f:
        pickle.dump(results_for_saving, f)
    
    print("All models and results saved successfully!")

# Main function
def main():
    """Main function to run the training pipeline"""
    print("=== Wine Quality Classification - Model Training ===")
    
    # Load and preprocess data
    df = load_dataset()
    print(f"Dataset loaded: {df.shape}")
    print(f"Features: {list(df.columns)}")
    
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    print(f"Data preprocessed. Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train and evaluate models
    results, trained_models = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Save models and results
    save_models_and_results(results, scaler)
    
    # Print final results
    print("\n=== Final Results ===")
    print("{:<20} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"))
    print("-" * 80)
    
    for name, result in results.items():
        print("{:<20} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            name,
            result['accuracy'],
            result['auc'],
            result['precision'],
            result['recall'],
            result['f1'],
            result['mcc']
        ))
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
    print(f"\nBest Model: {best_model} (Accuracy: {results[best_model]['accuracy']:.4f})")

if __name__ == "__main__":
    main()
