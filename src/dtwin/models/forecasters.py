
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression

FEATURE_GROUPS = {
    "cgm": lambda col: col.startswith("cgm_"),
    "meals": lambda col: col.startswith("meal_"),
    "wearable": lambda col: col.startswith("wearable_") or col.startswith("tod_"),
    "ehr": lambda col: col.startswith("ehr_"),
    "twin_state": lambda col: col.startswith("twin_")
}

def filter_features_by_groups(all_cols: list[str], allowed_groups: list[str]) -> list[str]:
    selected = []
    for col in all_cols:
        for grp in allowed_groups:
            if grp in FEATURE_GROUPS and FEATURE_GROUPS[grp](col):
                selected.append(col)
                break
    return selected

class BaseForecaster:
    def fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None = None):
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

class B0TrendForecaster(BaseForecaster):
    """B0: Linear trend extrapolation based on current glucose level and 30-min slope."""
    def fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None = None):
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Extrapolate: proj_g = current_g + slope_30m * 60 (for 60-120 min)
        curr_g = X.get("cgm_glucose_mgdl", pd.Series(100.0, index=X.index)).fillna(100.0)
        slope = X.get("cgm_slope_30m", pd.Series(0.0, index=X.index)).fillna(0.0)
        proj_peak = curr_g + slope * 60.0
        # Map projected peak to probability using sigmoid
        z = (proj_peak - 180.0) / 20.0
        prob = 1.0 / (1.0 + np.exp(-z))
        return np.clip(prob, 0.001, 0.999).values

class LightGBMFusionForecaster(BaseForecaster):
    def __init__(self, feature_groups: list[str], seed: int = 42):
        self.feature_groups = feature_groups
        self.seed = seed
        self.clf = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=seed,
            verbose=-1
        )
        self.selected_cols: list[str] = []
        self.calibrator: IsotonicRegression | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None = None):
        self.selected_cols = filter_features_by_groups(list(X.columns), self.feature_groups)
        if not self.selected_cols:
            self.selected_cols = list(X.columns)

        X_sub = X[self.selected_cols].copy()
        # Handle categoricals
        for col in X_sub.select_dtypes(include=["object", "category"]).columns:
            X_sub[col] = X_sub[col].astype("category")

        self.clf.fit(X_sub, y)

        # Fit isotonic calibrator on predictions
        raw_preds = self.clf.predict_proba(X_sub)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(raw_preds, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_sub = X[self.selected_cols].copy()
        for col in X_sub.select_dtypes(include=["object", "category"]).columns:
            X_sub[col] = X_sub[col].astype("category")

        raw_preds = self.clf.predict_proba(X_sub)[:, 1]
        if self.calibrator is not None:
            return self.calibrator.transform(raw_preds)
        return raw_preds
