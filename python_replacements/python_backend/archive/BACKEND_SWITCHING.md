> **HISTORICAL — This document describes planning/design context for the Python-only backend cutover. It is not current implementation guidance. See `python_backend/BACKEND_CUTOVER.md` and `python_backend/ALGO_PARITY.md` for the current status.**

# BACKEND_SWITCHING

**Purpose:** Define how a Python-native backend can replace the Julia backend *in practice*, given the constraint that `PySR_custom` is treated as the consumer contract.

---

## 1) Hard truth: “Python-only” requires a switching mechanism

Today, `PySR_custom` imports and uses `juliacall`/`juliapkg` to call Julia code. If `PySR_custom` is *truly* immutable (no code changes, no env-var hooks, no monkeypatch), then a **Python-only** runtime is not achievable: some component must still provide the Julia interface it expects.

So we must choose one of these strategies and document it explicitly.

---

## 2) Strategy options

### A) Minimal change to `PySR_custom` (recommended for true Python-only)

Add a backend selector (env var or init kwarg) to route calls to:

- `JuliaBackend` (current)
- `PythonBackend` (new)

This is the only clean way to achieve “no Julia runtime” without hacks. It does violate “unchanged PySR_custom”, but can be done as a **small, auditable, backward-compatible** change.

**Status:** Best technical outcome; conflicts with strict “PySR_custom unchanged”.

---

### B) Replace `juliacall` with a Python stub module (no PySR_custom edits, but fragile)

Arrange `PYTHONPATH` so `import juliacall` resolves to *our* package first, implementing:

- the subset of the `juliacall` API that `PySR_custom` uses
- `Main` / `seval` behaviors needed by `PySR_custom`

Pros:
- `PySR_custom` source unchanged.

Cons:
- Very brittle (must mimic an external library’s API).
- Any upstream `PySR_custom` change can break the stub.
- Not recommended unless you control deployment tightly.

**Status:** Possible; high maintenance risk.

---

### C) Keep Julia runtime, but move “algorithm” to Python (not Python-only)

Use a tiny Julia shim package (`SymbolicRegression.jl`) that forwards calls into Python via `PythonCall.jl`.

Pros:
- `PySR_custom` unchanged.
- Easy to interop with existing Julia call sites.

Cons:
- Still requires Julia at runtime → not “Python-only”.

**Status:** Good transitional step; not the stated end goal.

---

### D) Freeze current behavior and ship a self-contained Julia runtime (still not Python-only)

Vendor Julia + the minimal Julia backend and treat it as an internal dependency.

Pros:
- Robust; minimal new engineering.

Cons:
- Still Julia at runtime.

**Status:** Operationally practical; not Python-only.

---

## 3) Recommendation

If the end goal is **actually no Julia runtime**, adopt Strategy A (small backend selector in `PySR_custom`).

If the immediate goal is to begin Python backend work while keeping `PySR_custom` source untouched, Strategy C is a reasonable interim approach (Julia shim calling Python), but it is not the final state.

---

## 4) Next decision needed

Pick one:

1. **Strict**: no edits to `PySR_custom` ever (then accept “not Python-only”, or accept fragile stubbing).
2. **Pragmatic**: allow a tiny, backward-compatible backend selector in `PySR_custom` (recommended).

