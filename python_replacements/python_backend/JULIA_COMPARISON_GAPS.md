# Julia Comparison Gaps (Requires Julia Installation)

These validation items require a working Julia installation with
SymbolicRegression.jl. They cannot be completed in the current
Julia-free development environment.

## Algorithm Parity

- Compare mutation distribution (form_connection, break_connection, etc.)
  against Julia over many sampled mutations with identical random seeds.
- Compare crossover behavior on equivalent parent trees between Python
  and Julia search implementations.
- Compare regularized evolution population dynamics: acceptance rates,
  tournament pressure, age histograms.
- Compare HOF replacement and aging behavior — Python backend uses a
  simpler HOF implementation than Julia's full featured one.
- Compare constant optimization outcomes for nonlinear formulas with
  multiple constants — Julia uses Optim.jl while Python uses SciPy.
- Compare operator selection frequencies under equal and skewed weights.
- Compare parsimony pressure effects on expression size distributions.
- Compare model-selection outputs (best, accuracy, score) quantitatively.
- Check whether Python discovers equivalent symbolic forms or only
  numerically close approximations.
- Investigate systematic expression size bias between backends.

## Real Dataset Benchmarks

- Run 5–10 representative real datasets through both backends.
- Compare: loss, complexity, runtime, reproducibility, equation stability.
- Establish acceptance thresholds for real workloads.
- Track best/median/worst outcomes across multiple seeds.
- Validate extrapolation behavior on held-out test data.

## Julia-Only Features

- `expression_spec` templates — implemented as Julia macros.
- Custom operator definitions through Julia AST manipulation.
- Julia-native loss functions (need Julia for definition).
- Multi-node distributed search.

## How to Run

```sh
# Install Julia and SymbolicRegression.jl
julia -e 'using Pkg; Pkg.add("SymbolicRegression")'

# Use backend="julia" to select the real Julia backend
PYSR_BACKEND=julia python -c "
from pysr import PySRRegressor
import numpy as np
X = np.random.randn(100, 3)
y = X[:, 0]**2
model = PySRRegressor(niterations=10)
model.fit(X, y)
print(model.equations_)
"
```

Compare output with:
```sh
# Python backend
PYSR_BACKEND=python python -c "
from pysr import PySRRegressor
import numpy as np
X = np.random.randn(100, 3)
y = X[:, 0]**2
model = PySRRegressor(niterations=10, backend='python')
model.fit(X, y)
print(model.equations_)
"
```
