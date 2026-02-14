"""Decision Tree Model"""
from sklearn.tree import DecisionTreeClassifier
import sys
sys.path.append('..')
from utils import BaseModel

class DecisionTreeModel(BaseModel):
    def __init__(self):
        super().__init__("Decision Tree")
        self.model = DecisionTreeClassifier(random_state=42)
