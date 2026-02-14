"""Random Forest Model (Ensemble)"""
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append('..')
from utils import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__("Random Forest")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
