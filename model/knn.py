"""K-Nearest Neighbors Model"""
from sklearn.neighbors import KNeighborsClassifier
import sys
sys.path.append('..')
from utils import BaseModel

class KNNModel(BaseModel):
    def __init__(self):
        super().__init__("kNN")
        self.model = KNeighborsClassifier(n_neighbors=5)
