# Python Replacements For The Julia Backend

This folder contains the Python components that replace the Julia runtime layer.

## Contents

- `python_backend/` — the native symbolic-regression backend implemented in Python.
- `juliacall/` — a local compatibility shim that satisfies PySR imports and bridges expected Julia-facing calls into Python behavior where needed.

## How This Connects To PySR

`PySR_custom` is the user-facing API. It accepts familiar `PySRRegressor` options, validates them, then dispatches supported search work into `python_backend`.

The connection is:

```text
PySRRegressor.fit(...)
    -> adjusted PySR backend selector
    -> BackendOptions construction
    -> python_backend.backend.PythonSRBackend.equation_search(...)
    -> Python Hall of Fame / best equations / artifacts
    -> PySR-style equations_, predict(), get_best(), exports
```

The `juliacall` shim exists because upstream PySR imports and some compatibility surfaces expect a `juliacall` module. In this project, that shim prevents Julia from being a runtime requirement.

## Python Backend Responsibilities

`python_backend` implements the Julia-replacement behavior, including:

- expression trees and canonical serialization
- operator registry and numpy evaluation
- regularized evolution search
- mutation and crossover
- Hall of Fame tracking
- constant optimization
- constraints and nested constraints
- weights and custom losses
- batching
- warm start and multi-population search
- multi-output search
- artifacts, trace data, digests, and validation
- option contract checking
- dimensional checks
- Eureqa-style math operators
- row-history operators: `delay`, `sma`, `wma`, `mma`, `median`

## Operator Extension Model

New operators are wired through these places:

1. `python_backend/ops.py` — token, arity, numpy function, out-buffer function.
2. `python_backend/data/sr-operator-registry.yaml` — registry metadata.
3. `python_backend/expr.py` — parsing and SymPy export.
4. `python_backend/dimensional.py` — dimensional behavior.
5. `python_backend/gradients.py` — constant-optimization gradients or zero-gradient behavior.
6. `python_backend/capabilities.py` — allowlist/capability support.
7. `python_backend/tests/` — evaluation/export/search tests.

## Runtime Usage

Install/copy both `PySR_custom` and these replacement packages into the same environment, then force the Python backend:

```bash
export PYSR_BACKEND=python
```

Python usage remains PySR-style:

```python
from pysr import PySRRegressor

model = PySRRegressor(
    niterations=20,
    binary_operators=["+", "-", "*", "/", "min", "max"],
    unary_operators=["sin", "cos", "gauss", "erf", "tanh"],
)
model.fit(X, y)
```

For row-history operators, preserve row order and avoid treating rows as exchangeable samples:

```python
model = PySRRegressor(
    binary_operators=["+", "*", "sma", "delay"],
    unary_operators=[],
)
```

The second argument to history operators is rounded to a positive integer window size.

## Validation

Run from the project root that contains `python_backend`:

```bash
python3 -m pytest python_backend/tests -q
```

Expected packaged baseline at the time this folder was created:

```text
388 passed, 0 failed
```
