import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo
from src.lmnn import LMNN
from src.fblmnn import FeasibleLMNN


def fetch_data(id):
    dataset = fetch_ucirepo(id=id)

    # Extract features and targets (assuming they are stored as pandas objects)
    X = dataset.data.features
    y = dataset.data.targets

    # Convert to NumPy arrays if necessary.
    # This ensures that X becomes a 2D array (n_samples, n_features)
    # and y becomes a 1D array (n_samples,).
    if hasattr(X, "values"):
        X = X.values
    else:
        X = np.array(X)

    if hasattr(y, "values"):
        y = y.values
    else:
        y = np.array(y)

    # Ensure y is a 1D array (avoiding column-vector issues)
    y = y.ravel()
    return (X, y)


def evaluate_lmnn_methods():
    datasets = {}

    # X_wine, y_wine = fetch_data(id=109)
    # datasets["wine"] = (X_wine, y_wine)

    X_iris, y_iris = fetch_data(id=53)
    datasets["iris"] = (X_iris, y_iris)

    X_bc, y_bc = fetch_data(id=17)
    datasets["breast_cancer"] = (X_bc, y_bc)

    # X_balance, y_balance = fetch_data(id=12)
    # datasets["iris"] = (X_balance, y_balance)
    #
    results = {}

    for dataset_name, (X, y) in datasets.items():
        print(f"\nEvaluating dataset: {dataset_name}")
        results[dataset_name] = {}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        acc = knn.score(X_test, y_test)
        results[dataset_name]["knn"] = acc

        sp_lmnn = LMNN(k=3, max_iter=1000, tol=1e-5)
        sp_lmnn.fit(X_train, y_train)
        X_train_sp = sp_lmnn.transform(X_train)
        X_test_sp = sp_lmnn.transform(X_test)
        knn_sp = KNeighborsClassifier(n_neighbors=3)
        knn_sp.fit(X_train_sp, y_train)
        acc_sp = knn_sp.score(X_test_sp, y_test)
        results[dataset_name]["sp_lmnn"] = acc_sp

        try:
            mp_lmnn = LMNN(k=3, max_iter=1000, tol=1e-5)
            mp_lmnn.fit(X_train, y_train, mode="multi", outer_iter=5)
        except TypeError:
            mp_lmnn = LMNN(k=3, max_iter=1000, tol=1e-5)
            mp_lmnn.fit(X_train, y_train)
        X_train_mp = mp_lmnn.transform(X_train)
        X_test_mp = mp_lmnn.transform(X_test)
        knn_mp = KNeighborsClassifier(n_neighbors=3)
        knn_mp.fit(X_train_mp, y_train)
        acc_mp = knn_mp.score(X_test_mp, y_test)
        results[dataset_name]["mp_lmnn"] = acc_mp

        fb_lmnn = FeasibleLMNN(
            k=3, regularization=0.5, gamma=1.0, margin=1.0, max_iter=1000, tol=1e-5
        )
        fb_lmnn.fit(X_train, y_train)
        X_train_fb = fb_lmnn.transform(X_train)
        X_test_fb = fb_lmnn.transform(X_test)
        knn_fb = KNeighborsClassifier(n_neighbors=3)
        knn_fb.fit(X_train_fb, y_train)
        acc_fb = knn_fb.score(X_test_fb, y_test)
        results[dataset_name]["fb_lmnn"] = acc_fb

        print(f"  knn: {acc*100:.2f}%")
        print(f"  sp-lmnn: {acc_sp*100:.2f}%")
        print(f"  mp-lmnn: {acc_mp*100:.2f}%")
        print(f"  fb-lmnn: {acc_fb*100:.2f}%")

    return results


if __name__ == "__main__":
    evaluate_lmnn_methods()
