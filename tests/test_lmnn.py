import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import unittest
from src.lmnn import LMNN

class TestLMNN(unittest.TestCase):
    def setUp(self):
        # Load a small dataset
        iris = load_iris()
        self.X = iris.data[:10]  # Use only first 10 samples for quick testing
        self.y = iris.target[:10]
        
        # Scale the data
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)
        
        # Initialize LMNN
        self.lmnn = LMNN(k=3, max_iter=10)  # Small number of iterations for testing
    
    def test_initialization(self):
        """Test if LMNN initializes with correct parameters"""
        self.assertEqual(self.lmnn.k, 3)
        self.assertEqual(self.lmnn.max_iter, 10)
        
    def test_fit_transform(self):
        """Test if fit_transform works and returns correct shape"""
        X_transformed = self.lmnn.fit_transform(self.X_scaled, self.y)
        self.assertEqual(X_transformed.shape, self.X_scaled.shape)
        
    def test_transform(self):
        """Test if transform works after fitting"""
        self.lmnn.fit(self.X_scaled, self.y)
        X_transformed = self.lmnn.transform(self.X_scaled)
        self.assertEqual(X_transformed.shape, self.X_scaled.shape)

if __name__ == '__main__':
    unittest.main() 