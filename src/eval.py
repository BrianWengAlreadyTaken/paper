import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Optional
import tabulate

from src.data import load_datasets
from src.lmnn import LMNN
from src.fblmnn import FeasibleLMNN
from src.utils import time_model
from src.drlmnn import DRLMNN
from src.klmnn import KernelLMNN
from src.mmlmnn import MultiMetricLMNN


@time_model
def evaluate_knn(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Evaluate standard KNN."""
    print("evaluate_knn")
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


@time_model
def evaluate_sp_lmnn(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Evaluate SP-LMNN (single-pass LMNN)."""
    print("evaluate_sp_lmnn")
    lmnn = LMNN(k=3, max_iter=200, tol=1e-5)
    lmnn.fit(X_train, y_train)
    X_train_t = lmnn.transform(X_train)
    X_test_t = lmnn.transform(X_test)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_t, y_train)
    return model.score(X_test_t, y_test)


@time_model
def evaluate_mp_lmnn(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Evaluate MP-LMNN (multi-pass LMNN)."""
    print("evaluate_mp_lmnn")
    lmnn = LMNN(k=3, max_iter=200, tol=1e-5, passes=5)
    try:
        lmnn.fit(X_train, y_train)
    except TypeError:
        lmnn.fit(X_train, y_train)
    X_train_t = lmnn.transform(X_train)
    X_test_t = lmnn.transform(X_test)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_t, y_train)
    return model.score(X_test_t, y_test)


@time_model
def evaluate_fb_lmnn(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Evaluate Feasible LMNN."""
    print("evaluate_fb_lmnn")
    lmnn = FeasibleLMNN(
        k=3, regularization=0.5, gamma=1.0, margin=1.0, max_iter=200, tol=1e-5
    )
    lmnn.fit(X_train, y_train)
    X_train_t = lmnn.transform(X_train)
    X_test_t = lmnn.transform(X_test)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_t, y_train)
    return model.score(X_test_t, y_test)


@time_model
def evaluate_dr_lmnn(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Evaluate DR-LMNN (Dimensionality Reduction LMNN)."""
    print("evaluate_dr_lmnn")
    n_features = X_train.shape[1]
    n_components = min(2, n_features)
    lmnn = DRLMNN(
        k=3, n_components=n_components, regularization=0.5, max_iter=200, tol=1e-5
    )
    lmnn.fit(X_train, y_train)
    X_train_t = lmnn.transform(X_train)
    X_test_t = lmnn.transform(X_test)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_t, y_train)
    return model.score(X_test_t, y_test)


@time_model
def evaluate_k_lmnn(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Evaluate K-LMNN (Kernelized LMNN)."""
    print("evaluate_k_lmnn")
    lmnn = KernelLMNN(k=3, regularization=0.5, gamma=1.0, max_iter=200, tol=1e-5)
    lmnn.fit(X_train, y_train)
    X_train_t = lmnn.transform(X_train)
    X_test_t = lmnn.transform(X_test)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_t, y_train)
    return model.score(X_test_t, y_test)


@time_model
def evaluate_mm_lmnn(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    print("evaluate_mm_lmnn")
    n_samples = X_train.shape[0]
    k = 3
    n_clusters = min(3, n_samples // (k + 1)) or 1
    lmnn = MultiMetricLMNN(
        k=3, n_clusters=n_clusters, regularization=0.5, max_iter=200, tol=1e-5
    )
    lmnn.fit(X_train, y_train)
    X_train_t = lmnn.transform(X_train)
    X_test_t = lmnn.transform(X_test)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_t, y_train)
    return model.score(X_test_t, y_test)


@time_model
def evaluate_svm(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Evaluate SVM (SVC with RBF kernel)."""
    print("evaluate_svm")
    model = SVC(kernel="rbf", random_state=42)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


def evaluate_dataset(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Evaluate all models on a single dataset."""
    print(f"\nEvaluating dataset: {name}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    results = {}
    models = {
        "knn": evaluate_knn,
        "sp_lmnn": evaluate_sp_lmnn,
        "mp_lmnn": evaluate_mp_lmnn,
        "fb_lmnn": evaluate_fb_lmnn,
        "dr_lmnn": evaluate_dr_lmnn,
        "k_lmnn": evaluate_k_lmnn,
        "mm_lmnn": evaluate_mm_lmnn,
        "svm": evaluate_svm,
    }

    for model_name, evaluate_fn in models.items():
        try:
            accuracy, time_taken = evaluate_fn(X_train, y_train, X_test, y_test)  # pyright: ignore
            print(f"{model_name}, {accuracy=}, {time_taken=}")
            results[model_name] = {"accuracy": accuracy, "time": time_taken}
        except Exception as e:
            print(f"Error evaluating {model_name} on {name}: {str(e)}")
            results[model_name] = {"accuracy": np.nan, "time": 0.0}

    return results


def run_experiments() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Run evaluation for all models on all datasets."""
    datasets = load_datasets()
    results = {}

    for name, (X, y) in datasets.items():
        results[name] = evaluate_dataset(name, X, y)
        format_results({name: results[name]})
    return results


def format_results(results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """Print results in a tabular format."""
    print("\nEvaluation Results:")
    for dataset_name, model_results in results.items():
        print(f"\nDataset: {dataset_name}")
        table = []
        headers = ["Model", "Accuracy (%)", "Time (s)"]
        for model_name, metrics in model_results.items():
            accuracy = metrics["accuracy"]
            time_taken = metrics["time"]
            table.append(
                [
                    model_name,
                    f"{accuracy * 100:.2f}" if not np.isnan(accuracy) else "N/A",
                    f"{time_taken:.4f}",
                ]
            )
        print(tabulate.tabulate(table, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    results = run_experiments()
