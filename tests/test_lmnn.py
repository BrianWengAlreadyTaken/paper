import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import euclidean_distances
import unittest
from src.lmnn import LMNN
import time

class TestLMNN(unittest.TestCase):
    def setUp(self):
        iris = load_iris()
        self.X = iris.data[:10]
        self.y = iris.target[:10]
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)
        self.lmnn = LMNN(k=3, max_iter=200, tol=1e-5)
    
    def test_initialization(self):
        self.assertEqual(self.lmnn.k, 3)
        self.assertEqual(self.lmnn.max_iter, 200)
        
    def test_fit_transform(self):
        X_transformed = self.lmnn.fit_transform(self.X_scaled, self.y)
        self.assertEqual(X_transformed.shape, self.X_scaled.shape)
        
    def test_transform(self):
        self.lmnn.fit(self.X_scaled, self.y)
        X_transformed = self.lmnn.transform(self.X_scaled)
        self.assertEqual(X_transformed.shape, self.X_scaled.shape)

    def test_accuracy_improvement(self):
        iris = load_iris()
        X = iris.data
        y = iris.target
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        n_samples = len(X)
        train_size = int(0.7 * n_samples)
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Original k-NN
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        y_pred_original = knn.predict(X_test)
        accuracy_original = accuracy_score(y_test, y_pred_original)
        
        # LMNN
        start_time = time.time()
        X_train_lmnn = self.lmnn.fit_transform(X_train, y_train)
        lmnn_time = time.time() - start_time
        
        # Distance metrics
        original_distances = euclidean_distances(X_train)
        original_avg = np.mean(original_distances)
        transformed_distances = euclidean_distances(X_train_lmnn)
        transformed_avg = np.mean(transformed_distances)
        
        X_test_lmnn = self.lmnn.transform(X_test)
        knn.fit(X_train_lmnn, y_train)
        y_pred_lmnn = knn.predict(X_test_lmnn)
        accuracy_lmnn = accuracy_score(y_test, y_pred_lmnn)
        
        # Print results
        print(f"\nResults:")
        print(f"Original k-NN accuracy: {accuracy_original:.4f}")
        print(f"LMNN k-NN accuracy: {accuracy_lmnn:.4f}")
        print(f"Improvement: {accuracy_lmnn - accuracy_original:+.4f}")
        print(f"Training time: {lmnn_time:.2f}s")
        print(f"Distance change: {(transformed_avg - original_avg)/original_avg:+.2%}")
        
        self.assertGreaterEqual(accuracy_lmnn, accuracy_original)

if __name__ == '__main__':
    unittest.main() 