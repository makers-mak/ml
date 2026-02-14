# Income Prediction System – BITS WILP ML Assignment 2

## Problem Statement

Build a complete machine learning classification pipeline to predict whether a person's annual income exceeds $50,000 based on census data. The solution implements six different classification algorithms, evaluates them using multiple performance metrics, and deploys an interactive web application for real-time predictions.

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

| Model Name                | Type      | Key Parameters              |
|---------------------------|-----------|-----------------------------|
| Logistic Regression       | Linear    | max_iter=1000               |
| Decision Tree             | Tree      | default depth               |
| k-Nearest Neighbors (kNN) | Instance  | k=5 neighbors               |
| Naive Bayes (Gaussian)    | Bayesian  | default parameters          |
| Random Forest (Ensemble)  | Bagging   | 100 trees                   |
| XGBoost (Ensemble)        | Boosting  | 100 estimators, logloss     |

## Performance Metrics

Each model is evaluated using six standard classification metrics:

| ML Model              | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|-----------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression   | X.XXXX   | X.XXXX | X.XXXX    | X.XXXX | X.XXXX | X.XXXX |
| Decision Tree         | X.XXXX   | X.XXXX | X.XXXX    | X.XXXX | X.XXXX | X.XXXX |
| kNN                   | X.XXXX   | X.XXXX | X.XXXX    | X.XXXX | X.XXXX | X.XXXX |
| Naive Bayes           | X.XXXX   | X.XXXX | X.XXXX    | X.XXXX | X.XXXX | X.XXXX |
| Random Forest         | X.XXXX   | X.XXXX | X.XXXX    | X.XXXX | X.XXXX | X.XXXX |
| XGBoost               | X.XXXX   | X.XXXX | X.XXXX    | X.XXXX | X.XXXX | X.XXXX |

> **Note**: Run `python train_models_simple.py` to generate actual metrics. They will be saved to `model/metrics_summary.csv`.

## Model Performance Observations

| Model               | Observations                                                                                                                                                                                                              |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression | Provides a solid baseline with good interpretability. Works well for this linearly separable problem. Fast training and prediction times make it suitable for real-time applications.                                    |
| Decision Tree       | Captures non-linear patterns but may overfit on training data. Performance varies based on tree depth. Provides clear decision rules that are easy to explain to non-technical stakeholders.                              |
| kNN                 | Performance depends heavily on the choice of k and distance metric. Sensitive to feature scaling. Computationally expensive for large datasets but no training phase required.                                           |
| Naive Bayes         | Fast and efficient despite independence assumption. Works surprisingly well even when features are correlated. Handles categorical features naturally. Good for scenarios requiring probabilistic predictions.            |
| Random Forest       | Ensemble approach reduces overfitting compared to single decision tree. Robust to outliers and handles mixed feature types well. Provides feature importance rankings. Strong candidate for production deployment.        |
| XGBoost             | Typically achieves the best performance through gradient boosting. Handles class imbalance effectively. Includes regularization to prevent overfitting. Industry-standard choice for tabular data classification tasks. |

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
python train_models_simple.py
```

This script will:
- Download the Adult Income dataset from UCI
- Clean and encode the data
- Train all 6 classification models
- Save models to `model/` directory
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

### 3. Run Web Application

```bash
streamlit run app_simple.py
```

The app will open at `http://localhost:8501`

### 4. Use the Application

1. **View Model Comparison**: See performance metrics for all 6 models
2. **Select Model**: Choose from dropdown in sidebar
3. **Upload Test Data**: CSV file with same columns as training data
4. **Get Predictions**: Click "Run Prediction" to see results
5. **View Metrics**: If target column included, see performance evaluation

## Project Structure

```
ML/streamlit/
├── app_simple.py              # Streamlit web application
├── train_models_simple.py     # Model training pipeline
├── requirements.txt           # Python dependencies
├── README_ADULT_INCOME.md    # This file
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── label_encoders.pkl    # For categorical encoding
│   └── metrics_summary.csv    # Model performance metrics
└── data/
    └── (dataset downloaded automatically)
```

## Key Features

### Training Pipeline (`train_models_simple.py`)
- ✅ Automatic dataset download from UCI
- ✅ Data cleaning and validation
- ✅ Simple label encoding for categorical features
- ✅ Stratified train-test split (80/20)
- ✅ All 6 models trained consistently
- ✅ Comprehensive metric calculation
- ✅ Model persistence with metadata

### Web Application (`app_simple.py`)
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


## Sample Test Data Format

Your CSV file should have these columns:

```csv
age,workclass,fnlwgt,education,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country,income
39, State-gov,77516, Bachelors,13, Never-married, Adm-clerical, Not-in-family, White, Male,2174,0,40, United-States,<=50K
50, Self-emp-not-inc,83311, Bachelors,13, Married-civ-spouse, Exec-managerial, Husband, White, Male,0,0,13, United-States,<=50K
```

**Note**: The 'income' column is optional. Include it only if you want performance evaluation.

## Deployment

### Streamlit Community Cloud

1. Push code to GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repository and branch
5. Set main file: `app.py`
6. Click "Deploy"

## License & Acknowledgments

- **Dataset**: UCI Machine Learning Repository
- **Citation**: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.

## Author

**Machine Learning Assignment 2**  
Adult Income Classification System

---

