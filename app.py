"""
ML for Multiple Models
- Command line: python app.py --train (trains all models)
- Web app: streamlit run app.py (serves predictions)
"""
import os
import sys
import pandas as pd
import streamlit as st
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Import model modules
from model import ALL_MODELS, MODEL_FILES
from utils import load_and_prepare_data, BaseModel


MODEL_DIR = "model"
TARGET = 'income' # Target Column y

def train_all_models():
    """Train all 6 models - Command line mode"""
    print("\n" + "="*60)
    print("TRAINING ALL MODELS - Adult Income Classification")
    print("="*60 + "\n")
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, label_encoders, feature_names = load_and_prepare_data()
    
    # Save label encoders
    joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
    print("Label encoders saved.\n")
    
    # Train all models
    results = []
    for name, ModelClass in ALL_MODELS.items():
        model_instance = ModelClass()
        model_instance.train(X_train, y_train)
        metrics = model_instance.evaluate(X_test, y_test)
        results.append(metrics)
        
        # Save model
        filename = MODEL_FILES[name]
        filepath = os.path.join(MODEL_DIR, filename)
        model_instance.save(filepath, feature_names)
    
    # Save metrics summary
    metrics_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("METRICS SUMMARY")
    print("="*60)
    print(metrics_df.to_string(index=False))
    
    metrics_df.to_csv(os.path.join(MODEL_DIR, "metrics_summary.csv"), index=False)
    print(f"\nMetrics saved to {MODEL_DIR}/metrics_summary.csv")
    print("\n✓ Training complete! All models saved.\n")

def load_model(model_name):
    """Load a trained model"""
    filepath = os.path.join(MODEL_DIR, MODEL_FILES[model_name])
    return BaseModel.load(filepath)

def load_encoders():
    """Load label encoders"""
    return joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))

def preprocess_data(df, encoders):
    """Apply label encoding to categorical columns"""
    df_encoded = df.copy()
    for col, encoder in encoders.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(
                lambda x: x if x in encoder.classes_ else encoder.classes_[0]
            )
            df_encoded[col] = encoder.transform(df_encoded[col])
    return df_encoded

def run_streamlit_app():
    """Run Streamlit web application"""
    st.set_page_config(page_title="Income Prediction using Multiple Models", layout="wide")
    
    st.title("Predict Adult Income - with ML Models")
    st.markdown("Predict if a person's income exceeds $50K based on census data")
    
    # Check if models are trained
    metrics_path = os.path.join(MODEL_DIR, "metrics_summary.csv")
    if not os.path.exists(metrics_path):
        st.error("⚠️ Models not trained yet!")
        st.info("Please run: `python app.py --train` to train models first.")
        return
    
    # Model Performance Overview
    st.header("📊 Model Performance Overview")
    try:
        metrics_df = pd.read_csv(metrics_path)
        st.dataframe(
            metrics_df.style.highlight_max(
                axis=0,
                subset=['Accuracy', 'AUC', 'F1'],
                color='lightgreen'
            ),
            use_container_width=True
        )
        
        # Visual comparison
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            metrics_df.plot(x='Model', y=['Accuracy', 'F1'], kind='bar', ax=ax, rot=45)
            ax.set_ylabel('Score')
            ax.set_title('Comparing Model Performance')
            ax.legend(loc='lower right')
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            best_acc = metrics_df.loc[metrics_df['Accuracy'].idxmax(), 'Model']
            best_f1 = metrics_df.loc[metrics_df['F1'].idxmax(), 'Model']
            
            st.markdown("### Top Performed Metrics")
            st.info(f"**Best Accuracy:** {best_acc}")
            st.info(f"**Best F1 Score:** {best_f1}")
            
            st.markdown("### Chosen Dataset Details")
            st.write("- **Source:** UCI Adult Income")
            st.write("- **Task:** Binary Classification")
            st.write("- **Classes:** ≤50K / >50K")
    
    except Exception as e:
        st.warning(f"Could not load metrics: {e}")
    
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Select Model to Evaluate")
    selected_model = st.sidebar.selectbox("Choose Model:", list(MODEL_FILES.keys()))

    from utils import COLUMN_NAMES
    COLUMN_NAMES_STR = ", ".join(COLUMN_NAMES)
    st.sidebar.markdown(f"""
    ### 📋 Instructions
    1. Upload a CSV file with similar features of dataset: **UCI Adult Income**
    2. Expected column headers in dataset csv: `{COLUMN_NAMES_STR}`
    3. We can **download the sample test_data.csv** clicking below button, then **upload** in rightside content region and **test predictions by switching models** above
    4. Uploaded Data file must contain the same columns as training data, and click on **Run Prediction** button.
    5. Include target column 'income' for evaluation (optional)
    6. Note: The code is designed and implemented for robust, fully working end‑to‑end functionality.
    """)
    # Show download button only if file exists
    SAMPLE_TEST_PATH = os.path.join("test_data.csv")
    if os.path.exists(SAMPLE_TEST_PATH):
        sample_df = pd.read_csv(SAMPLE_TEST_PATH, sep=";")  # adjust sep based on data
        st.sidebar.download_button(
            label="⬇️ Download sample test_data.csv",
            data=sample_df.to_csv(index=False, sep=";"),
            file_name="test_data.csv",
            mime="text/csv",
        )
    else:
        st.sidebar.warning("Sample test_data.csv not found in the project folder. You can download from UCI and add headers manually")
    
    # File Upload
    st.header(" Upload Test Data")
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file:
        # Smart CSV loading - handles any format
        try:
            # Try reading with default settings
            test_df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
        except:
            try:
                # Try with semicolon separator
                test_df = pd.read_csv(uploaded_file, sep=';')
                uploaded_file.seek(0)
            except:
                st.error("❌ Could not read CSV file. Please check the format.")
                st.stop()
        
        # Auto-detect if headers are missing (check if columns are generic like "Unnamed")
        if any('Unnamed' in str(col) for col in test_df.columns):
            from utils import COLUMN_NAMES
            uploaded_file.seek(0)
            test_df = pd.read_csv(uploaded_file, names=COLUMN_NAMES, na_values=' ?', skipinitialspace=True)
        
        # If no header but has 15 numeric columns, assume it's Adult dataset without headers
        if len(test_df.columns) == 15 and all(pd.api.types.is_numeric_dtype(test_df[col]) or test_df[col].dtype == 'object' for col in test_df.columns):
            if not any(str(col).lower() in ['age', 'income', 'education'] for col in test_df.columns):
                from utils import COLUMN_NAMES
                uploaded_file.seek(0)
                test_df = pd.read_csv(uploaded_file, names=COLUMN_NAMES, na_values=' ?', skipinitialspace=True)
        
        st.subheader("📋 Data Preview")
        st.dataframe(test_df.head(10))
        
        # Show data info
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", len(test_df))
        col2.metric("Columns", len(test_df.columns))
        col3.metric("Missing Values", test_df.isnull().sum().sum())
        
        # Load model to get expected features
        model_data = load_model(selected_model)
        expected_features = model_data['feature_names']
        
        # Check if target column exists
        has_target = TARGET in test_df.columns
        
        if has_target:
            X_test = test_df.drop(columns=[TARGET])
            y_true = test_df[TARGET].map({'>50K': 1, '<=50K': 0, 1: 1, 0: 0})
            st.success(f"✓ Target column '{TARGET}' found. Full evaluation will be shown.")
        else:
            X_test = test_df.copy()
            y_true = None
            st.warning(f"⚠️ Target column '{TARGET}' not found. Showing predictions only.")
        
        # Check for missing features
        missing_features = set(expected_features) - set(X_test.columns)
        extra_features = set(X_test.columns) - set(expected_features)
        
        if missing_features:
            st.warning(f"⚠️ Missing features: {', '.join(missing_features)}")
            st.info("The model was trained on different features. Predictions may not be accurate.")
        
        if extra_features:
            st.info(f"ℹ️ Extra columns will be ignored: {', '.join(extra_features)}")
            
            from utils import COLUMN_NAMES
            missing_cols = [c for c in COLUMN_NAMES if c not in X_test.columns]
            if missing_cols:
                st.error(
                    "Uploaded file is missing expected columns:\n\n"
                    + ", ".join(missing_cols)
                    + "\n\nPlease upload a CSV with the same headers as the training dataset. Cross check if headers missing in dataset.\n\n"
                    + "Click **Download sample test_data.csv** button, to get the **adult income dataset with headers** added."
                )
                st.stop()
            
            X_test = X_test[expected_features]
        
        if st.button(" Run Prediction"):
            with st.spinner(f"Running {selected_model}..."):
                try:
                    model = model_data['model']
                    encoders = load_encoders()
                    
                    # Preprocess data
                    X_encoded = preprocess_data(X_test, encoders)
                    
                    # Ensure column order matches training
                    X_encoded = X_encoded[expected_features]
                    
                    predictions = model.predict(X_encoded)
                    pred_proba = model.predict_proba(X_encoded)[:, 1] if hasattr(model, 'predict_proba') else None
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.info("Please ensure your data matches the training format.")
                    st.stop()
                
                st.subheader("🎯 Predictions")
                result_df = pd.DataFrame({
                    'Predicted Income': ['> $50K' if p == 1 else '≤ $50K' for p in predictions[:10]],
                    'Probability (>50K)': [f"{p:.2%}" for p in pred_proba[:10]] if pred_proba is not None else ['N/A'] * 10
                })
                st.dataframe(result_df)
                
                if has_target:
                    st.subheader("📈 Performance Metrics")
                    
                    acc = accuracy_score(y_true, predictions)
                    prec = precision_score(y_true, predictions, zero_division=0)
                    rec = recall_score(y_true, predictions, zero_division=0)
                    f1 = f1_score(y_true, predictions, zero_division=0)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{acc:.4f}")
                    col2.metric("Precision", f"{prec:.4f}")
                    col3.metric("Recall", f"{rec:.4f}")
                    col4.metric("F1 Score", f"{f1:.4f}")
                    
                    if pred_proba is not None:
                        auc = roc_auc_score(y_true, pred_proba)
                        mcc = matthews_corrcoef(y_true, predictions)
                        
                        col5, col6 = st.columns(2)
                        col5.metric("AUC", f"{auc:.4f}")
                        col6.metric("MCC", f"{mcc:.4f}")
                    
                    st.subheader("🔢 Confusion Matrix")
                    cm = confusion_matrix(y_true, predictions)
                    
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title(f'Confusion Matrix - {selected_model}')
                    st.pyplot(fig)
                    
                    with st.expander("**View Detailed Classification Report**", expanded=True):
                        report = classification_report(
                            y_true, predictions,
                            target_names=['≤50K', '>50K'],
                            zero_division=0
                        )
                        st.text(report)
    else:
        st.info("👆 Upload a CSV file to get started. We can download sample file by clicking **Download sample test_data.csv** button on left sidebar.")
    
    st.markdown("---")
    st.markdown("**ML Models Training & Metrics** | Adult Income Classification")

# Main entry point
if __name__ == "__main__":
    # Check if running in training mode
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        train_all_models()
    else:
        # Run Streamlit app
        run_streamlit_app()
