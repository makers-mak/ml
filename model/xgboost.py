"""XGBoost Model (Ensemble)"""
from xgboost import XGBClassifier
import sys
sys.path.append('..')
from utils import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self):
        super().__init__("XGBoost")
        self.model = XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
