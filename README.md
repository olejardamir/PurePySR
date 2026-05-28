# PySR Custom Python-Only Symbolic Regression

This project packages the work needed to run `PySR_custom` symbolic regression without a Julia runtime.

The original PySR architecture expects Julia, `juliacall`, and `SymbolicRegression.jl` to execute the search backend. This project exists to preserve the PySR-style Python API while replacing the Julia execution layer with a native Python backend that can be developed further for product use.

## Folder Layout

- `pysr_adjusted/` — copied PySR files that were adjusted so PySR can select and call the Python backend instead of requiring Julia.
- `python_replacements/` — copied Python replacement packages that provide the symbolic-regression backend and the local `juliacall` compatibility shim.
- `CONTRACT_PYSR_CUSTOM.md` — option-support contract for the Python-only backend.
- `TECHNICAL_PARITY_MATRIX.md` — generated parity/status matrix for supported, rejected, and ignored PySR options.

## What This Enables

- Run symbolic regression from Python without installing Julia.
- Keep the familiar `PySRRegressor`-style API.
- Use the Python backend for equation search, operators, constraints, Hall of Fame handling, warm start, artifacts, and validation.
- Continue backend development in Python, including additional math operators and future product-specific features.

## Current Scope

Implemented scope includes numeric symbolic regression, Python-only backend dispatch, Julia compatibility shims, extended Eureqa-style math operators, and row-history operators such as `delay`, `sma`, `wma`, `mma`, and moving `median`.

Not included as a product promise:

- exact byte-for-byte equivalence with Julia/SymbolicRegression.jl
- Julia distributed execution
- Julia JIT/turbo/Bumper behavior
- Julia macro expression templates
- arbitrary user-defined Python/string-manipulation functions

## Runtime Principle

Production/runtime code should use the Python backend. Julia should be treated only as an optional reference tool for parity experiments, not as a required dependency.

Typical runtime setting:

```bash
export PYSR_BACKEND=python
```

Then use PySR normally from Python:

```python
from pysr import PySRRegressor

model = PySRRegressor(
    niterations=20,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "exp", "log", "gauss", "erf"],
)
model.fit(X, y)
```

## Validation Status At Packaging Time

The copied source was validated from the original working tree with:

```bash
python3 -m pytest python_backend/tests -q
```

Result at packaging time:

```text
388 passed, 0 failed
```

Warnings are expected for documented ignored/rejected Julia-only options and numerical edge-case tests.
