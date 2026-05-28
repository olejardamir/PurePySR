# API Decisions — Python-Only Backend

## Autodiff backend

**Decision: Rejected at PySR API level, supported at backend level.**

`autodiff_backend` raises `ValueError` in PySR_custom before the backend is
reached (`sr.py` line ~1680). The backend itself (`constant_optimization.py`)
supports `autodiff_backend=True` in `optimize_constants()` and it is tested
(`validate_deep.py::test_direct_backend_autodiff_convergence`).

Rationale for keeping it rejected at the PySR API level:
- Autodiff through the Julia backend used ForwardDiff.jl, which has no
  Python equivalent. Our pure-Python autodiff (finite-difference-based)
  is much slower and only useful for constant optimization, not full
  search.
- Exposing it to users would create an incorrect expectation of Julia-level
  performance.
- If a true autodiff library (e.g., JAX, Aesara) is integrated later,
  this decision can be revisited.

## Expression spec

**Decision: Rejected.**

`expression_spec` templates are a Julia macro feature with no Python
equivalent. Users who need nonstandard expression representations should
use the standard `binary_operators`/`unary_operators` interface.

## Custom operator definitions (inline)

**Decision: Rejected.**

PySR supports defining custom operators inline with
`extra_sympy_mappings`. The Python backend does not support this because
it requires sympy expression parsing that is not implemented.

The option gate classifies these as REJECTED; the existing xfail test
was converted to an expected rejection (ValueError raised before search).

## Julia-only accepted-but-ignored options

**Decision: Keep as accepted-but-ignored with warnings.**

Options like `turbo`, `fast_cycle`, `parallelism`, `procs`, `bumper`
are Julia-specific and have no Python equivalent. They emit a single
`UserWarning` with a clear explanation. This is preferred over hard
rejection because:
- Users migrating from Julia-backed PySR should get one clear warning
  per run, not a fatal error.
- The option gate contract explicitly marks these as IGNORED so the
  contract is transparent.

If silent semantic drift is discovered (an option changes behavior
unexpectedly without warning), that option should be reclassified to
REJECTED.

## Backend selection

**Decision: Expose `backend="python"` and `backend="julia"` for compatibility.**

`backend="python"` selects the Python backend only. `backend="julia"` uses
the juliacall shim (which delegates to the Python backend). Both work
identically in the current codebase because no real Julia is available.

No `backend="auto"` or automatic detection — users must explicitly choose.
If real Julia + SymbolicRegression.jl is installed, the user must set
`PYSR_BACKEND=julia` to activate real Julia interop; otherwise the shim
is used regardless of the backend parameter value.

## API compatibility philosophy

**Decision: Prioritize PySR drop-in behavior for supported options.**

The Python backend aims to be a drop-in replacement for PySR on the
SUPPORTED option set. Options marked REJECTED fail with clear error
messages. Options marked IGNORED emit warnings.

The `TECHNICAL_PARITY_MATRIX.md` document defines exactly what is
supported, rejected, or ignored. Any option not listed there is unknown
and raises a hard error before search starts.

## Final decisions (from correctness/robustness audit)

After deep correctness validation (27 tests), robustness testing (35 tests), and
strengthened tree-inspection tests (23 tests), all 334 tests pass. No semantic
drift was found in any currently supported option.

| Item | Decision | Rationale |
|------|----------|-----------|
| autodiff_backend | REJECTED at API, supported at backend | No true autodiff lib; FD-based only for const opt |
| expression_spec | PERMANENTLY UNSUPPORTED | Julia macro feature, no Python equivalent |
| custom operators inline | REJECTED | No sympy parsing; ValueError before search |
| backend="julia" | Delegates to Python shim | Safe fallback when no real Julia |
| Julia-only IGNORED | Keep warnings | One clear warning per option, not fatal |
| protected_div | No changes needed | Correctly handles 1/0 as inf, not crash |
| safe_log | No changes needed | Handles 0, negative correctly |
| power negative base | Already invalid | Caught by numeric safety, finite predictions |

## Out-of-scope features (intentionally not ported)

These PySR features are intentionally out-of-scope for the Python-only
backend and will not be implemented unless explicitly requested:

- Julia-specific parallel execution (multithreading, distributed)
- GPU/TPU acceleration
- JIT compilation (LoopVectorization, etc.)
- Bumper.jl memory management
- Julia-native expression macros
- Full Julia ecosystem package integration
- Real juliacall interop (the shim is a compatibility layer only)
