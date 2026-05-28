"""Minimal PySR example: fit a symbolic expression, print equations, predict.

No Julia required — uses the pure-Python backend (default).
"""

import numpy as np
from pysr import PySRRegressor

# Generate synthetic data: y = x0^2 + 0.5 * x1
rng = np.random.default_rng(42)
X = rng.standard_normal((100, 3)).astype(np.float64)
y = (X[:, 0] ** 2 + X[:, 1] * 0.5).astype(np.float64)

model = PySRRegressor(
    niterations=5,
    population_size=20,
    binary_operators=["+", "-", "*"],
    unary_operators=[],
    verbosity=1,
)

model.fit(X, y)

# Show the Pareto-front equations
print(model.equations_)

# Predict on new data
X_new = rng.standard_normal((5, 3)).astype(np.float64)
preds = model.predict(X_new)
print("Predictions:", preds)
