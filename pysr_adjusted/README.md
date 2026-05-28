# Adjusted PySR Files Only

This folder intentionally contains only the `PySR_custom` files that were changed or removed to bypass Julia and route PySR into the Python-only backend.

It is not a full PySR checkout. It is an instruction/reproduction bundle showing exactly which PySR-side files need to be adjusted.

## Layout

- `files/` — adjusted files copied from `PySR_custom`, preserving their original relative paths.
- `deleted_files/FILES_REMOVED_TO_BYPASS_JULIA.txt` — files deleted from `PySR_custom` as part of removing Julia runtime/package requirements.

## Adjusted Files

The adjusted files currently included are:

```text
README.md
docs/src/backend.md
docs/src/examples.md
environment.yml
pyproject.toml
pysr/julia_extensions.py
pysr/julia_import.py
pysr/sr.py
```

The removed Julia-specific files are listed in:

```text
deleted_files/FILES_REMOVED_TO_BYPASS_JULIA.txt
```

## What These Adjustments Do

The important PySR-side changes are:

1. **Backend selection**
   - `pysr/sr.py` adds/uses backend selection so runtime can use the Python backend instead of Julia.
   - `PYSR_BACKEND=python` is the intended runtime mode.

2. **Python backend dispatch**
   - `pysr/sr.py` maps PySRRegressor options into `python_backend.options.BackendOptions`.
   - Search is routed to `python_backend.backend.PythonSRBackend.equation_search`.
   - Backend results are converted back into PySR-style `equations_`, `get_best()`, `predict()`, SymPy export, and lambda export behavior.

3. **Julia import boundary**
   - `pysr/julia_import.py` and `pysr/julia_extensions.py` were adjusted so imports can work with the local `juliacall` compatibility shim instead of requiring real Julia.

4. **Julia package dependency removal**
   - `pyproject.toml` and `environment.yml` were adjusted to remove hard Julia runtime/package assumptions.
   - `pysr/juliapkg.json` and `pysr/julia_registry_helpers.py` were removed because the Python-only runtime must not resolve or install Julia packages.

5. **Explicit unsupported-feature behavior**
   - Julia-only features such as `expression_spec` and inline Julia-style custom operators are rejected clearly before search.
   - Julia-only runtime/performance behavior is either rejected or documented as ignored according to the option contract.

## How To Reproduce In Another PySR_custom Checkout

From a clean PySR_custom checkout:

1. Copy each file from `files/` over the same relative path in the PySR checkout.
2. Delete every path listed in `deleted_files/FILES_REMOVED_TO_BYPASS_JULIA.txt`.
3. Put `python_replacements/python_backend` and `python_replacements/juliacall` on the install path or package them with the project.
4. Set:

```bash
export PYSR_BACKEND=python
```

5. Run the backend tests from the project containing `python_backend`:

```bash
python3 -m pytest python_backend/tests -q
```

Expected packaged baseline:

```text
388 passed, 0 failed
```

## Important Boundary

These PySR files are only the adapter layer. The actual Julia replacement implementation lives in:

```text
../python_replacements/python_backend
../python_replacements/juliacall
```
