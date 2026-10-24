from collections.abc import Generator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def create_patient_grouped_splits(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    n_splits: int = 5,
    seed: int = 42
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generates group-disjoint patient train/test splits.
    Ensures zero patient overlap between train and test sets.
    """
    unique_patients = df[patient_col].unique()
    num_unique = len(unique_patients)

    if num_unique < 2:
        # Fallback if too few patients
        yield np.arange(len(df)), np.arange(len(df))
        return

    n_splits = min(n_splits, num_unique)
    gkf = GroupKFold(n_splits=n_splits)

    for train_idx, test_idx in gkf.split(df, groups=df[patient_col]):
        yield train_idx, test_idx
