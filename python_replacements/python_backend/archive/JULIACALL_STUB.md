> **HISTORICAL — This document describes planning/design context for the Python-only backend cutover. It is not current implementation guidance. See `python_backend/BACKEND_CUTOVER.md` and `python_backend/ALGO_PARITY.md` for the current status.**

# juliacall Stub (Repo-Local)

This repository can shadow the third-party `juliacall` Python package with a
local stub implementation to enable a **Julia-free** runtime path while keeping
`PySR_custom/` unchanged.

## How it works

If the repo root appears first on `PYTHONPATH`, `import juliacall` will resolve
to `./juliacall/` in this repository instead of the real package.

The stub implements only a minimal subset of the API used by `PySR_custom`:

- `juliacall.Main` with `seval(...)`
- `juliacall.convert(...)`
- `SymbolicRegression.Options(...)` / `SymbolicRegression.equation_search(...)`
  that delegates to `python_backend.PythonSRBackend`

### Artifact outputs

After `equation_search(...)` returns, the stub writes governed artifacts into
the run directory (`{output_directory}/{run_id}/`):

| file | source | format |
|---|---|---|
| `trace.jsonl` | `result['trace_records']` | JSONL via `dump_jsonl()` |
| `digests.json` | `result['digests']` | stable-key JSON |
| `policy.json` | `result['policy_dict']` | stable-key JSON |
| `dataset.json` | `result['dataset_manifest']` | stable-key JSON + `dataset_digest` |
| `archive.json` | `result['hall_of_fame']` | canonical JSON via `canonical_json()` |
| `hall_of_fame.csv` | `result['hall_of_fame']` | CSV for PySR_custom `_read_equation_file()` |
| `checkpoint.pkl` | stub constant | pickle for PySR_custom warm-start checks |

These files match the layout produced by `cli_run_golden.py` (via
`run_and_write_artifacts`) so they can be diffed and validated with
`cli_validate_trace.py`.  Example:

```sh
python python_backend/cli_validate_trace.py \
    --trace output_dir/run_id/trace.jsonl \
    --digests output_dir/run_id/digests.json
```

The `canonical_json()` serializer from `python_backend/trace.py` is used for
`archive.json` so that digests match across reference runs.

## No Julia binary required

The stub intercepts every call site inside PySR_custom that would normally
talk to the Julia runtime.  A test (`test_stub_no_julia_binary_invoked` in
`test_juliacall_stub_integration.py`) proves this by placing a fake ``julia``
executable on PATH that exits nonzero and prints a distinctive marker.  The
test asserts the marker never appears — if it did, PySR_custom would have
shelled out to a real Julia.

This is the closest we can get to proving "no real Julia invoked" without a
full system-call audit.  Any future regression where PySR_custom adds a new
`subprocess.run("julia ...")` call will cause this test to fail.

## Enable

From the repo root:

1) Ensure repo root is first on `PYTHONPATH`:
   ```sh
   PYTHONPATH=$PWD:$PYTHONPATH
   ```
2) Import and run `PySR_custom` normally.
3) Artifacts can be located under the `output_directory` model parameter.

## Risks / caveats

- This is **fragile** and **security-sensitive**: it intentionally shadows a
  third-party package name. Use only in controlled environments.
- The stub does not implement the full JuliaCall API. If `PySR_custom` begins
  using additional Julia features, the stub must be extended.
- The stub is intended for SR-L2/SR-L3 paving; it is not a drop-in replacement
  for real Julia-backed PySR.

