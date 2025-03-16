import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from lmnn import LMNN

# Load and prepare the data
# Let's try a more challenging dataset
data = load_breast_cancer()  # Changed from Iris to Breast Cancer dataset
X, y = data.data, data.target

# Split the data with a larger test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train LMNN with different parameters
lmnn = LMNN(k=3, 
            regularization=0.5, 
            learning_rate=1e-7,
            max_iter=1000)

print("Training LMNN...")
X_train_lmnn = lmnn.fit_transform(X_train_scaled, y_train)
X_test_lmnn = lmnn.transform(X_test_scaled)

# Compare kNN with and without LMNN using cross-validation
print("\nPerforming cross-validation...")

# Without LMNN
knn = KNeighborsClassifier(n_neighbors=3)
cv_scores_original = cross_val_score(knn, X_train_scaled, y_train, cv=5)
knn.fit(X_train_scaled, y_train)
y_pred_original = knn.predict(X_test_scaled)

# With LMNN
knn_lmnn = KNeighborsClassifier(n_neighbors=3)
cv_scores_lmnn = cross_val_score(knn_lmnn, X_train_lmnn, y_train, cv=5)
knn_lmnn.fit(X_train_lmnn, y_train)
y_pred_lmnn = knn_lmnn.predict(X_test_lmnn)

# Print detailed results
print("\nResults without LMNN:")
print(f"Cross-validation scores: {cv_scores_original}")
print(f"Mean CV accuracy: {cv_scores_original.mean():.4f} (+/- {cv_scores_original.std() * 2:.4f})")
print(f"Test set accuracy: {accuracy_score(y_test, y_pred_original):.4f}")
print("\nClassification Report without LMNN:")
print(classification_report(y_test, y_pred_original))

print("\nResults with LMNN:")
print(f"Cross-validation scores: {cv_scores_lmnn}")
print(f"Mean CV accuracy: {cv_scores_lmnn.mean():.4f} (+/- {cv_scores_lmnn.std() * 2:.4f})")
print(f"Test set accuracy: {accuracy_score(y_test, y_pred_lmnn):.4f}")
print("\nClassification Report with LMNN:")
print(classification_report(y_test, y_pred_lmnn)) 