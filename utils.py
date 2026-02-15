"""
Utility functions for ML models
Includes: Data loading, Base model class, and helper functions
"""
import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

# Configuration
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
COLUMN_NAMES = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]
TARGET = 'income' # Target column 
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_and_prepare_data():
    """Load Adult Income dataset and prepare for training"""
    print("Loading Adult Income dataset...")
    df = pd.read_csv(DATA_URL, names=COLUMN_NAMES, na_values=' ?', skipinitialspace=True)
    
    print(f"Dataset shape: {df.shape}")
    if df.shape[0] < 500:
        raise ValueError("Dataset must have at least 500 instances")
    if df.shape[1] - 1 < 12:
        raise ValueError("Dataset must have at least 12 features")
    
    df = df.dropna()
    print(f"After removing missing values: {df.shape}")
    
    df[TARGET] = df[TARGET].map({'>50K': 1, '<=50K': 0})
    
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    print("Encoding categorical features...")
    categorical_cols = X.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Class distribution: {y.value_counts().to_dict()}\n")
    
    return X_train, X_test, y_train, y_test, label_encoders, X.columns.tolist()

class BaseModel:
    """Base class for all ML models"""
    def __init__(self, name):
        self.name = name
        self.model = None
        
    def train(self, X_train, y_train):
        """Train the model"""
        print(f"Training {self.name}...")
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        """Make predictions"""
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_test)[:, 1]
        return None
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        metrics = {
            'Model': self.name,
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
            'F1': round(f1_score(y_test, y_pred, zero_division=0), 4),
            'MCC': round(matthews_corrcoef(y_test, y_pred), 4)
        }
        
        if y_proba is not None:
            metrics['AUC'] = round(roc_auc_score(y_test, y_proba), 4)
        else:
            metrics['AUC'] = 0.0
            
        print(f"  Accuracy: {metrics['Accuracy']} | AUC: {metrics['AUC']} | F1: {metrics['F1']}")
        return metrics
    
    def save(self, filepath, feature_names):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'feature_names': feature_names,
            'name': self.name
        }
        joblib.dump(model_data, filepath)
        print(f"  Saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load model from disk"""
        return joblib.load(filepath)
