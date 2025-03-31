# Large Margin Nearest Neighbor (LMNN) Implementation

This repository contains a Python implementation of the Large Margin Nearest Neighbor (LMNN) algorithm as described in the paper "Distance Metric Learning for Large Margin Nearest Neighbor Classification" by Weinberger, Blitzer, and Saul.

## Installation

1. Clone the repository:

```bash
git clone [your-repository-url]
cd lmnn-project
```

2. Create a virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate.bat

# Unix/MacOS
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Basic usage example:

```python
from src.lmnn import LMNN
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and prepare data
data = load_breast_cancer()
X, y = data.data, data.target

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train LMNN
lmnn = LMNN(k=3, regularization=0.5)
X_train_lmnn = lmnn.fit_transform(X_train_scaled, y_train)
X_test_lmnn = lmnn.transform(X_test_scaled)
```

For a complete example, see `examples/example.py`.

## Parameters

The LMNN class accepts the following parameters:

- `k` (int, default=3): Number of target neighbors
- `regularization` (float, default=0.5): Regularization parameter for the optimization
- `learning_rate` (float, default=1e-7): Learning rate for gradient descent
- `max_iter` (int, default=1000): Maximum number of iterations
- `tol` (float, default=1e-5): Convergence tolerance

## Project Structure

```
lmnn-project/
│
├── src/               # Source code
│   ├── __init__.py
│   └── lmnn.py
│
├── examples/          # Example scripts
│   └── example.py
│
├── tests/             # Test files
│   └── test_lmnn.py
│
├── requirements.txt   # Project dependencies
├── setup.py          # Package setup file
└── README.md         # Project documentation
```

## Experiment Result

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@inproceedings{weinberger2006distance,
  title={Distance metric learning for large margin nearest neighbor classification},
  author={Weinberger, Kilian Q and Blitzer, John and Saul, Lawrence K},
  booktitle={Advances in neural information processing systems},
  pages={1473--1480},
  year={2006}
}
```
