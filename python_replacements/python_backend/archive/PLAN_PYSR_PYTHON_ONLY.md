> **HISTORICAL — This document describes planning/design context for the Python-only backend cutover. It is not current implementation guidance. See `python_backend/BACKEND_CUTOVER.md` and `python_backend/ALGO_PARITY.md` for the current status.**

# PySR_custom → Python-only Backend Plan

**Goal:** Remove the Julia runtime dependency by reimplementing the `SymbolicRegression.jl` backend behavior in Python, while keeping **`PySR_custom` unchanged** (as long as feasible). Julia remains the reference implementation until parity is proven by tests.

**Non-goal:** Mechanical `.jl` → `.py` translation. This is a reimplementation with staged compatibility and regression testing.

---

## 0) Ground Rules (Compatibility Contract)

1. **`PySR_custom` is the contract.** The Python-facing API, parameter validation, and data formats used by `PySR_custom` must continue to work.
2. **Julia backend remains the reference** until the Python backend matches behavior within declared tolerances.
3. **Determinism is first-class:** fixed seeds, stable tie-breaks, explicit numeric policies, and reproducible reductions.
4. **Scope is feature-gated by evidence:** we only remove/replace features after tests demonstrate they are unused or matched.

Deliverable: a short “contract” document mapping `PySR_custom` knobs → required backend semantics.

---

## 1) Spec & Semantics Extraction (from EquationCode)

Inputs (primary):

- `EquationCode/CUSTOM/EQC-SR-v1.0.3-final.md`
- `EquationCode/BRIDGE.md` (change governance + trace requirements)
- `EquationCode/EQC.md` (semantics blocks, determinism, ordering)
- `EquationCode/FIC/*` (workflow from pseudocode → implementable components)

Tasks:

1. Extract **normative semantics** for:
   - objective comparison, tie-break rules, invalid (NaN/Inf) handling
   - randomness locality + PRNG policy
   - numeric policy (dtype, stability, rounding assumptions)
2. Produce an **Operator Manifest** for the SR algorithm components we must implement (initialization, mutation, selection, evaluation, acceptance, recording).
3. Define **trace schema** (minimal logs/metrics needed to compare Julia vs Python).

Deliverables:

- `SPEC_PYSR_BACKEND.md` (normative “what it means”)
- `TRACE_SCHEMA.md` (what to log and how to compare)

---

## 2) Observed-Behavior Inventory (from PySR_custom)

Tasks:

1. Enumerate the **exact Julia entrypoints** used by `PySR_custom`:
   - `Options(...)`, `equation_search(...)`, serialization paths, returned objects.
2. Enumerate **parameter surface area**:
   - what `PySR_custom` passes into Julia (operators, constraints, loss functions, batching, warm-start, progress/logging, migration/parallelism, template/parametric specs, units, etc.).
3. Identify file outputs that `PySR_custom` expects (hall-of-fame CSV, logs, saved state).

Deliverable:

- `CONTRACT_PYSR_CUSTOM.md` (table: parameter → backend requirement → test coverage)

---

## 3) Dual-Backend Harness (Julia reference vs Python candidate)

Strategy:

- Build a runner that can execute the same run config in:
  1) Julia backend (current state, via `juliacall`)
  2) Python backend (new package/module)
- Normalize outputs into a common representation and compare.

Comparison levels (choose explicitly per metric):

- **BITWISE** (rare; only for integer/structural outputs)
- **TOLERANCE** (floating metrics like loss/score)
- **DISTRIBUTIONAL** (if parallelism causes non-deterministic ordering; avoid if possible)

Deliverables:

- `tests/` (or `validation/`) with:
  - golden configs (small, fast)
  - golden traces (for regression)
  - drift report tool (what changed, why)

---

## 4) Python Backend Architecture (Incremental Implementation)

### 4.1 Data model

Implement Python equivalents of the minimal runtime structures:

- Expression tree representation (`Node`, `Expression`, operator enum/mapping)
- Dataset/batching representation
- Population / hall-of-fame storage
- Option/config object with validation matching Julia expectations

### 4.2 Core loop (first milestone)

Milestone A: **serial, single-output, float64**, basic operators, basic losses.

- generate initial population
- mutation operators (subset first)
- selection + tournament
- evaluation (vectorized where possible)
- record hall-of-fame
- return results in the shape `PySR_custom` expects

### 4.3 Performance strategy (after correctness)

Options (choose based on profiling):

- NumPy vectorization + caching
- Numba for hot loops (tree eval, mutation)
- Optional JAX/Torch for accelerated eval (only if needed)

Deliverable:

- `pysr_backend_py/` (or similar) with a stable API surface.

---

## 5) Migration Plan (How PySR_custom swaps backends)

Preferred approach (no `PySR_custom` changes):

- Provide a drop-in replacement module/package that `juliacall`-using code can bypass via environment variable (if already supported), or by ensuring `SymbolicRegression` calls route to Python backend through a thin compatibility layer.

If `PySR_custom` truly cannot be changed at all:

- Keep Julia import path but replace Julia backend with a minimal Julia “shim” that forwards to Python (least desirable), or
- Maintain Julia runtime but move algorithm to Python (not Python-only).

Decision gate:

- Confirm whether `PySR_custom` can accept a new backend selector without breaking the contract.

Deliverable:

- `BACKEND_SWITCHING.md` (exact mechanism + constraints)

---

## 6) Test Matrix (must-pass before removing Julia)

Minimum must-pass suite:

1. **Smoke:** tiny dataset, 2 iterations, serial, deterministic seed.
2. **Operators:** unary/binary sets, custom operator parsing.
3. **Constraints:** maxsize/minsize, nested constraints, complexity penalties.
4. **Losses:** standard and custom expression loss (if used by PySR_custom).
5. **Serialization:** save/load state equivalence (if exposed/used).
6. **Templates/Parametric:** only if `PySR_custom` enables them.
7. **Units:** only if `PySR_custom` enables them.

For each test: define expected invariants (e.g., monotonic best loss, stable equation formatting, deterministic hall-of-fame order).

---

## 7) De-Julia Cutover (Final Step)

Only after:

- Python backend passes the suite vs Julia reference across representative configs,
- performance is acceptable for your workloads,
- and a stable switching mechanism exists,

…then:

1. Disable Julia backend initialization in production path.
2. Remove Julia package resolution/dependency.
3. Keep a “reference mode” option for a while (Julia run for debugging) if desired.

Deliverable:

- release checklist + rollback plan.

---

## Immediate Next Action (recommended)

1. Read and summarize `EquationCode/CUSTOM/EQC-SR-v1.0.3-final.md` into:
   - required operators
   - determinism/tie-break rules
   - trace schema
2. Generate `SPEC_PYSR_BACKEND.md` and `TRACE_SCHEMA.md` drafts.

