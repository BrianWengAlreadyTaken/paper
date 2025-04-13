from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import load_wine, load_iris, load_breast_cancer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer


def clean_data(X, y) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert X and y to numerical NumPy arrays, handle missing values, ensure y is 1D.
    Returns (X, y) ready for train_test_split.
    """
    if hasattr(X, "values"):
        X = X.values
    else:
        X = np.array(X)

    if hasattr(y, "values"):
        y = y.values
    else:
        y = np.array(y)

    if isinstance(y, (pd.DataFrame, pd.Series)) and y.ndim > 1:
        y = y.iloc[:, 0]
    elif y.ndim > 1:
        y = y[:, 0]

    if y.dtype.kind in ["O", "U"]:
        y = LabelEncoder().fit_transform(y)

    X_df = pd.DataFrame(X)
    y_series = pd.Series(y)

    if X_df.isna().any().any():  # pyright: ignore
        imputer = SimpleImputer(strategy="mean")
        X_df = pd.DataFrame(imputer.fit_transform(X_df), columns=X_df.columns)

    if y_series.isna().any():
        valid_idx = ~y_series.isna()
        X_df = X_df.loc[valid_idx]
        y_series = y_series.loc[valid_idx]

    X = X_df.to_numpy(dtype=float)
    y = y_series.to_numpy(dtype=int)

    y = y.ravel()

    return X, y


def fetch_wine_data():
    try:
        dataset = fetch_ucirepo(id=109)
        X = dataset.data.features  # pyright: ignore
        y = dataset.data.targets  # pyright: ignore
    except Exception as e:
        print(f"UCI fetch failed for Wine (id=109): {str(e)}")
        print("Falling back to scikit-learn load_wine")
        data = load_wine()
        X = data.data  # pyright: ignore
        y = data.target  # pyright: ignore
    return clean_data(X, y)


def fetch_iris_data():
    try:
        dataset = fetch_ucirepo(id=53)
        X = dataset.data.features  # pyright: ignore
        y = dataset.data.targets  # pyright: ignore
    except Exception as e:
        print(f"UCI fetch failed for Iris (id=53): {str(e)}")
        print("Falling back to scikit-learn load_iris")
        data = load_iris()
        X = data.data  # pyright: ignore
        y = data.target  # pyright: ignore
    return clean_data(X, y)


def fetch_breast_cancer_data():
    try:
        dataset = fetch_ucirepo(id=17)
        X = dataset.data.features  # pyright: ignore
        y = dataset.data.targets  # pyright: ignore
    except Exception as e:
        print(f"UCI fetch failed for Breast Cancer (id=17): {str(e)}")
        print("Falling back to scikit-learn load_breast_cancer")
        data = load_breast_cancer()
        X = data.data  # pyright: ignore
        y = data.target  # pyright: ignore
    return clean_data(X, y)


def fetch_balance_data():
    try:
        dataset = fetch_ucirepo(id=12)
        X = dataset.data.features  # pyright: ignore
        y = dataset.data.targets  # pyright: ignore
    except Exception as e:
        print(f"UCI fetch failed for Balance Scale (id=12): {str(e)}")
        print("Falling back to OpenML")
        try:
            data = fetch_openml(data_id=997, as_frame=True)
            X = data.data  # pyright: ignore
            y = data.target  # pyright: ignore
            y = pd.Categorical(y).codes
        except Exception as e2:
            print(f"OpenML fetch failed: {str(e2)}")
            print("Skipping Balance Scale dataset")
            return None
    return clean_data(X, y)


def fetch_tic_tac_toe_data() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Fetch Tic-Tac-Toe dataset from UCI (id=101) or OpenML (id=50).
    Returns (X, y): numerical arrays (958, 9) and (958,).
    """
    try:
        dataset = fetch_ucirepo(id=101)
        X = dataset.data.features  # pyright: ignore
        y = dataset.data.targets  # pyright: ignore

        feature_mapping = {"x": 1, "o": 0, "b": 2}
        X = X.replace(feature_mapping)

        return clean_data(X, y)

    except Exception as e:
        print(f"UCI fetch failed for Tic-Tac-Toe (id=101): {str(e)}")
        print("Falling back to OpenML (data_id=50)")
        try:
            data = fetch_openml(data_id=50, as_frame=True, parser="pandas")
            X = data.data  # pyright: ignore
            y = data.target  # pyright: ignore

            feature_mapping = {"x": 1, "o": 0, "b": 2}
            X = X.replace(feature_mapping)

            return clean_data(X, y)

        except Exception as e2:
            print(f"OpenML fetch failed for Tic-Tac-Toe (id=50): {str(e2)}")
            print("Skipping Tic-Tac-Toe dataset")
            return None


def load_datasets() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load all datasets, skipping failed ones."""
    datasets = {}
    fetch_functions = {
        "wine": fetch_wine_data,
        "iris": fetch_iris_data,
        "breast_cancer": fetch_breast_cancer_data,
        "balance": fetch_balance_data,
        "tictactoe": fetch_tic_tac_toe_data,
    }

    for name, fetch_fn in fetch_functions.items():
        try:
            result = fetch_fn()
            if result is not None:
                datasets[name] = result
                print(
                    f"Loaded {name}: X.shape={result[0].shape}, classes={len(np.unique(result[1]))}"
                )
        except Exception as e:
            print(f"Unexpected error loading {name}: {str(e)}")
            print(f"Skipping {name} dataset")

    return datasets
