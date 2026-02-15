# Income Prediction System

## Problem Statement

Build a complete machine learning classification pipeline to predict whether a person's annual income exceeds $50,000 based on census data. The solution implements six different classification algorithms, evaluates them using multiple performance metrics, and deploys an interactive Streamlit web application for real-time predictions.

## Dataset Description

**Dataset**: UCI Adult Income (Census Income)  
**Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult)

- **Task**: Binary classification – predict if income >$50K or ≤$50K
- **Instances**: 48,842 records (after removing missing values: ~45,222)
- **Features**: 14 attributes including:
  - **Demographic**: age, race, sex, native country
  - **Work-related**: workclass, occupation, hours per week
  - **Education**: education level, education years
  - **Financial**: capital gain, capital loss
  - **Family**: marital status, relationship
- **Target Variable**: income (>50K = 1, ≤50K = 0)
- **Class Distribution**: ~75% earn ≤$50K, ~25% earn >$50K

## Models Implemented

All six models trained on the same 80/20 train-test split:

| ML Model Name             | Type      | Key Parameters              |
|---------------------------|-----------|-----------------------------|
| Logistic Regression       | Linear    | max_iter=1000               |
| Decision Tree             | Tree      | default depth               |
| k-Nearest Neighbors (kNN) | Instance  | k=5 neighbors               |
| Naive Bayes (Gaussian)    | Bayesian  | default parameters          |
| Random Forest (Ensemble)  | Bagging   | 100 trees                   |
| XGBoost (Ensemble)        | Boosting  | 100 estimators, logloss     |

## Performance Metrics

Each model is evaluated using six standard classification metrics, on chosen dataset (*100%):

| ML Model Name           | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|-------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression     | 0.8087   | 0.8210 | 0.7048    | 0.3533 | 0.4707 | 0.4021 |
| Decision Tree           | 0.8102   | 0.7519 | 0.5992    | 0.6397 | 0.6188 | 0.4931 |
| kNN                     | 0.7829   | 0.6780 | 0.5861    | 0.3342 | 0.4257 | 0.3222 |
| Naive Bayes             | 0.7998   | 0.8401 | 0.6803    | 0.3176 | 0.4330 | 0.3659 |
| Random Forest (Ensemble)| 0.8594   | 0.9108 | 0.7436    | 0.6346 | 0.6848 | 0.5981 |
| XGBoost (Ensemble)      | 0.8762   | 0.9286 | 0.7761    | 0.6830 | 0.7266 | 0.6492 |

> **Note**: Run `python app.py --train` to generate actual metrics, saved to `model/metrics_summary.csv`.

## Model Performance Observations

| ML Model Name            | Observation about model performance                                                                                                                                                                                                              |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression      | Provides a solid baseline with good interpretability. Works well for this linearly separable problem. Fast training and prediction times make it suitable for real-time applications.                                    |
| Decision Tree            | Captures non-linear patterns but may overfit on training data. Performance varies based on tree depth. Provides clear decision rules that are easy to explain to non-technical stakeholders. Good recall but lower precision                             |
| kNN                      | Performance (low) depends heavily on the choice of k and distance metric. Sensitive to feature scaling. Computationally expensive for large datasets but no training phase required.                                           |
| Naive Bayes              | Fast and efficient despite independence assumption (strong AUC). Works surprisingly well even when features are correlated. Handles categorical features naturally. Good for scenarios requiring probabilistic predictions.            |
| Random Forest (Ensemble) | Ensemble approach reduces overfitting compared to single decision tree. Robust to outliers and handles mixed feature types well. Provides feature importance rankings. Strong candidate for production deployment.       |
| XGBoost (Ensemble)       | Typically achieves the best performance through gradient boosting. Handles class imbalance effectively. Includes regularization to prevent overfitting. Industry-standard choice for tabular data classification tasks. Best overall: Highest Accuracy (87.62%), F1 (72.66%), AUC (92.86%).  |

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
python3 app.py --train
```

This script will:
- Download the Adult Income dataset from UCI
- Clean and encode the data
- Train all 6 classification models
- Save models pkl to `model/` directory
- Generate `model/metrics_summary.csv` with performance metrics

**Expected Output**:
```
Loading Adult Income dataset...
Dataset shape: (48842, 15)
After removing missing values: (45222, 15)
Encoding categorical features...

Training set: (36177, 14)
Test set: (9045, 14)

Training Logistic Regression...
  Accuracy: 0.XXXX | AUC: 0.XXXX | F1: 0.XXXX
  Saved to model/logistic_regression.pkl
...
✓ Training complete!
```

### 3. Launch Web Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. Note to check port.

### 4. Use the Application

1. **View Model Comparison**: See performance metrics for all 6 models
2. **Select Model**: Choose from dropdown in sidebar
3. **Upload Test Data**: CSV file with same columns as training data
4. **Get Predictions**: Click "Run Prediction" to see instant results
5. **View Metrics**: If target column included, see performance evaluation (Accuracy, F1, Confusion Matrix). Detailed precision/recall per class.

## Project Structure

```
ML-streamlit/
├── app.py                   # Streamlit web application + Training
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── utils.py                 # Data loading & BaseModel class
├── download_test_data.py    # Helper to download dataset to Train
├── model/
│   ├── logistic_regression.py   # Simple Python class and model initializer
│   ├── decision_tree.py
│   ├── knn.py
│   ├── naive_bayes.py
│   ├── random_forest.py
│   ├── xgboost.py
│   │
│   ├── logistic_regression.pkl  # models pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   │
│   ├── label_encoders.pkl   # For categorical encoding
│   └── metrics_summary.csv  # Model performance metrics
└── test_data.csv            # Sample for upload (dataset)
```

## Key Features

### Web Application (`app.py`)
- ✅ Clean, intuitive user interface
- ✅ Model performance comparison dashboard
- ✅ Interactive model selection
- ✅ CSV file upload with preview
- ✅ Real-time predictions
- ✅ Visual metrics display
- ✅ Confusion matrix visualization
- ✅ Detailed classification reports

## Technical Implementation

### Data Preprocessing
```python
# Simple label encoding approach
for col in categorical_columns:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col])
```

**Advantages**:
- Clean and straightforward
- No sparse matrices
- Works with all sklearn models
- Easy to understand and maintain

### Model Training Loop
```python
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Calculate metrics
    # Save model
```

**Benefits**:
- Uniform training process
- No special cases or complex logic
- Easy to add new models
- Clear and maintainable code

## Technologies Used

- **Python 3.8+**
- **scikit-learn**: ML models and metrics
- **XGBoost**: Gradient boosting
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation
- **Matplotlib & Seaborn**: Visualizations
- **Joblib**: Model serialization


## Sample Test Data Format: Adult Income dataset from UCI

Your CSV file should have these columns:

```csv
age,workclass,fnlwgt,education,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country,income
39, State-gov,77516, Bachelors,13, Never-married, Adm-clerical, Not-in-family, White, Male,2174,0,40, United-States,<=50K
50, Self-emp-not-inc,83311, Bachelors,13, Married-civ-spouse, Exec-managerial, Husband, White, Male,0,0,13, United-States,<=50K
```

**Note**: Include 'income', if we need performance evaluation.

## Deployment

### Streamlit Community Cloud

1. Push code to GitHub repository: https://github.com/makers-mak/ml.git
2. Visit [streamlit.io/cloud](https://share.streamlit.io)
3. New app > Select repository and branch
4. Set main file: `app.py` > Click "Deploy"
**Live Demo**: https://makers-ml.streamlit.app/

## License & Acknowledgments

- **Dataset**: UCI Machine Learning Repository
- **Citation**: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.
- **Author**: Adult Income Classification System

---
