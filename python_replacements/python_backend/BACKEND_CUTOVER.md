# Backend Cutover: Python‑only Transition (Complete)

**Status:** ✅ Complete — default backend is `"python"` (set in `sr.py:1068`).
Julia runtime dependency has been removed.

---

## 1) What exists now

| Component | Status |
|-----------|--------|
| `PythonSRBackend` (`backend.py`) | Full equation search loop in Python |
| `BackendAPI` protocol (`backend_api.py`) | Abstract interface for `equation_search` |
| `get_backend("python")` | Returns `PythonSRBackend` |
| `get_backend("julia")` | Raises `NotImplementedError` (stub) |
| Trace / artifact pipeline | `trace.jsonl`, `digests.json`, `policy.json`, `dataset.json`, `archive.json` written and validated |
| Validator CLI | Field‑level + digest‑level trace validation |
| `PySRRegressor._run_python_backend()` | Converts PySR params → `BackendOptions` → `PythonSRBackend.equation_search()` |
| `backend` kwarg + `PYSR_BACKEND` env var | Backend selector in `PySRRegressor.__init__` |

## 2) What was removed

| Artifact | Action |
|----------|--------|
| `SymbolicRegression.jl/` | Vendored Julia package — deleted |
| `julia_registry_helpers.py` | Dead code — deleted |
| `juliapkg.json` | No longer needed — deleted |
| `cli_run_pysr_stub_golden.py` | Stub CLI obsolete — deleted |
| `test_juliacall_stub_integration.py` | Stub integration tests obsolete — deleted |
| `juliacall` from `pyproject.toml` deps | Hard dependency removed (stub replaces it) |

## 3) What was kept (compatibility shims)

| Component | Purpose |
|-----------|---------|
| `juliacall/__init__.py` | Repo-local stub providing `Main`, `seval`, `AnyValue`, `VectorValue`, `SymbolicRegression` so PySR_custom imports succeed without Julia |
| `julia_import.py` | Routes to the stub when `PYSR_BACKEND=python` |
| `julia_helpers.py` | Bridge utilities (most are no-ops in Python mode) |
| `julia_extensions.py` | Minimal no-op stubs for `load_all_packages`, `load_required_packages` |
| `expression_specs.py`, `logger_specs.py` | API contracts — stub handles seval calls |
| `__init__.py` re‑exports (`jl`, `SymbolicRegression`) | Public API surface preserved |

## 4) Test results

| Metric | Count |
|--------|-------|
| Tests passing | **194 / 194** (0 failures) |
| Golden problems validated | **10 / 10** (all pass trace validation and acceptance criteria) |

## 5) Known limitations (future work)

| Limitation | Status |
|------------|--------|
| User‑defined lambda operators (e.g. `"(x) -> x^2"`) | Python backend supports only the predefined operator set |
| Julia backend path (`backend="julia"`) | Raises `NotImplementedError` (removed dependency) |
| `form_connection` / `break_connection` | Implemented (default weights 0.5 / 0.1) |

## 6) Running PySR_custom with the Python backend

No special configuration needed — `"python"` is the default:

```sh
python -c "
from pysr import PySRRegressor
model = PySRRegressor(...)  # uses Python backend by default
model.fit(X, y)
"
```

Or explicitly:
```sh
PYSR_BACKEND=python python -c "
from pysr import PySRRegressor
model = PySRRegressor(...)
model.fit(X, y)
"
```
