"""Model Package - All ML Models"""
from .logistic_regression import LogisticRegressionModel
from .decision_tree import DecisionTreeModel
from .knn import KNNModel
from .naive_bayes import NaiveBayesModel
from .random_forest import RandomForestModel
from .xgboost import XGBoostModel

# Model Registry
ALL_MODELS = {
    "Logistic Regression": LogisticRegressionModel,
    "Decision Tree": DecisionTreeModel,
    "kNN": KNNModel,
    "Naive Bayes": NaiveBayesModel,
    "Random Forest": RandomForestModel,
    "XGBoost": XGBoostModel
}

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

__all__ = [
    'LogisticRegressionModel',
    'DecisionTreeModel',
    'KNNModel',
    'NaiveBayesModel',
    'RandomForestModel',
    'XGBoostModel',
    'ALL_MODELS',
    'MODEL_FILES'
]
