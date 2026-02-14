"""Naive Bayes Model"""
from sklearn.naive_bayes import GaussianNB
import sys
sys.path.append('..')
from utils import BaseModel

class NaiveBayesModel(BaseModel):
    def __init__(self):
        super().__init__("Naive Bayes")
        self.model = GaussianNB()
