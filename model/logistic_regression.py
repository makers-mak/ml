"""Logistic Regression Model"""
from sklearn.linear_model import LogisticRegression
import sys
sys.path.append('..')
from utils import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__("Logistic Regression")
        self.model = LogisticRegression(max_iter=1000, random_state=42)
