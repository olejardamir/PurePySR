> **HISTORICAL — This document describes planning/design context for the Python-only backend cutover. It is not current implementation guidance. See `python_backend/BACKEND_CUTOVER.md` and `python_backend/ALGO_PARITY.md` for the current status.**

# SPEC_PYSR_BACKEND (Draft)

**Purpose:** Normative, implementation-oriented summary of `EQC-SR-v1.0.3` required semantics for building a **Python-native symbolic regression backend** that can eventually replace the Julia runtime, while keeping **`PySR_custom` as the fixed consumer contract**.

**Primary source:** `/home/glompy/Desktop/ASTROCYTECH/EquationCode/CUSTOM/EQC-SR-v1.0.3-final.md`

**Status:** Draft for review; intended to be split into subsystem specs later (EXPR/OPLIB/EVAL/RANK/SEARCH/TRACE).

---

## 1) Compliance & Claims

- Use EQC-SR compliance modes: `SR-DRAFT`, `SR-SPEC-READY`, `SR-PORT-READY`, `SR-RELEASE-READY`.
- All behavioral requirements that affect output, determinism, ranking, trace keys, or API results **MUST** be either:
  - `specified`, `not_applicable_with_reason`, `documented_deviation`, or `explicitly_deferred` (only if unreachable in target scope).
- Every compatibility claim must declare:
  - `compatibility_level` (SR-L0..SR-L5)
  - covered public surface + unsupported options
  - equivalence target + validation evidence
- Any option that is accepted but ignored must produce a **declared warning code** (never silently ignored).

Recommended initial target for a Python port (per EQC-SR): **SR-L2**, then SR-L3.

---

## 2) Global Objective & Ordering Semantics (REQ anchor: §4.1)

Default objective is a **vector**:

`objective(candidate) = (primary_loss, complexity, tie_break_key)`

Default comparison (deterministic total preorder):
1. lower `primary_loss` is better
2. if equal within `EPS_LOSS_EQ`, lower `complexity` is better
3. then deterministic `tie_break_key` (structural/canonical)
4. then lowest insertion index

Even in Pareto mode (partial order by loss/complexity), any step requiring a single best must define a deterministic fallback to a total order.

---

## 3) Invalid Candidate Policy (REQ anchor: §4.2)

Candidate is invalid if any mandatory validation fails, including:
- structural invalidity, arity mismatch, unresolved operator
- constraint violation (size/depth/nested constraints)
- evaluation failure not handled by protected operators
- non-finite objective under numeric policy

Invalid ranking:
- all valid candidates outrank all invalid candidates
- invalid candidates ordered deterministically by:
  1) `invalid_reason_code`
  2) structural key / canonical key
  3) insertion index

Invalids may be logged but must not enter hall-of-fame unless explicitly supported as a debug mode.

---

## 4) Reproducibility & Numeric Defaults (REQ anchor: §4.3)

Defaults (unless overridden by profile):
- seed space: `uint64`
- PRNG: PCG64 (or explicitly declared equivalent)
- floating dtype: IEEE-754 binary64
- fast-math: forbidden for governed runs
- stable sorts required
- argmin ties resolved deterministically
- NaN/Inf objective ranked worse than all finite valid objectives
- parallel reductions must be deterministic or disabled

The backend must be able to emit a replay token / run fingerprint sufficient to reproduce (at chosen equivalence level).

---

## 5) Expression Tree Contract (REQ anchors: §5.*)

### 5.1 Representation

Expression must be representable as a tree:

- `Expression := Node`
- `Node := Variable | Constant | OperatorCall`
- `OperatorCall := OperatorID(children...)`

Each node must expose (at least):
- kind, operator id (if any), arity, ordered children
- constant numeric value (if constant) and its equality policy
- variable index (if variable)
- size, depth, complexity
- canonical serialization (machine-stable)
- structural hash (derived from canonical serialization)

### 5.2 Identity & Equality

EQC-SR distinguishes object/structural/semantic/canonical identity.

Default duplicate detection and caching must be based on:
1) structural hash
2) canonical serialization equality (if hashes match)

Semantic equality is not used by default for duplicates.

### 5.3 Canonical Serialization (must be stable)

Canonical serialization must define:
- operator namespace + version encoding
- variable index encoding (default canonical forms are **0-based**)
- constant encoding (precision + rounding + stability)
- child order policy for commutative operators
- associative flattening policy
- unary-minus policy
- protected operator names
- version tag

Recommended canonical text style (example):
- `op.namespace.name_vN(child0,child1,...)`
- `var[0]`
- `const[float64:0x1.921fb54442d18p+1]`

Structural hash = `SHA256(canonical_expression_bytes)` by default.

---

## 6) Operator Registry Contract (REQ anchors: §6.*)

Backend must define an operator registry where each operator has a machine-readable record including:
- stable operator id / name
- arity
- category (protected math, user-defined, etc.)
- domain/codomain expectations (as needed)
- export mappings (e.g., SymPy/NumPy string forms)

Protected operator requirements:
- protected variants must be used or enforced so evaluation failure boundaries are deterministic
- failure behavior (division by 0, log of negative, overflow) must follow numeric policy and be reproducible

Custom operator policy:
- operators can be registered with explicit metadata
- string-based configuration must parse with a defined grammar (no ambiguous parsing)

---

## 7) Dataset & Input Contract (REQ anchors: §7.*)

Define a dataset record that makes evaluation deterministic:
- X shape conventions (feature axis vs row axis must be explicit)
- missing/non-finite input policy
- batching policy (how batches are sampled, and how loss aggregates)
- train/validation/test semantics if used
- feature/target type semantics (float vs integer vs categorical) must be declared

---

## 8) Evaluation Semantics (REQ anchors: §8.*)

Must define:
- candidate evaluation mapping: `(expression, X_batch) -> yhat`
- non-finite prediction policy (NaN/Inf predictions)
- evaluation output shapes and dtypes (deterministic)
- eligibility and semantics for any “compiled” backend
- evaluation cache contract (keying, invalidation, determinism)
- explicit boundary between “protected math returns finite” vs “evaluation failure”

---

## 9) Loss, Complexity, Ranking (REQ anchors: §9.*)

Must define:
- primary loss function(s) (e.g., MSE) and how weights/batching interact
- complexity scoring function (stable under export)
- score computation and ranking (esp. how to compare candidates and update archive)
- Pareto frontier computation + deterministic ordering of the frontier
- model selection contract (how a single “best” is selected if needed)

---

## 10) Search: Generation, Mutation, Crossover, Evolution (REQ anchors: §10–13)

Must define:
- random expression generator contract
- initial population contract
- seed derivation contract (substreams)
- mutation metadata requirements + validity rules
- crossover lineage contract (if implemented)
- selection contract (tournament etc.)
- evolution step skeleton (control flow only; operators define semantics)
- replacement policy
- termination contract

Note: EQC-SR recommends excluding broad distributed execution until golden tests pass.

---

## 11) Migration, Hall-of-Fame, Archive (REQ anchors: §14–15)

Must define:
- multi-population migration semantics if exposed
- hall-of-fame entry schema
- deterministic update rule and duplicate handling
- “equation table” contract (what gets reported/exported)
- archive immutability/revision rules

---

## 12) Constants & Optimization (REQ anchors: §16.*)

Must define:
- constant representation (ephemeral constants, precision, mutation)
- constant optimization contract (if enabled): deterministic behavior requirements, eligibility, and failure handling

---

## 13) Constraints, Simplification, Export (REQ anchors: §17–19)

Must define:
- structural constraints (size, depth, operator constraints)
- nested constraints grammar/representation
- dimensional/unit constraints (if used)
- constraint failure codes (stable registry)
- simplification vs canonicalization difference; when each runs
- export contract + export equivalence tests (e.g., SymPy/NumPy callable)

---

## 14) Equivalence Targets (REQ anchors: §22)

EQC-SR defines equivalence levels; pick one per validation suite:
- E0-SR: trace equivalent
- E1-SR: metric equivalent
- E2-SR: distribution equivalent
- E3-SR: invariant equivalent

For early Python backend work, prefer E1/E3 first, then tighten.

---

## 15) What this document does *not* decide (yet)

- Exact mapping from every `PySR_custom` option → EQC-SR capability profile (this becomes `CONTRACT_PYSR_CUSTOM.md`).
- Which parallelism modes are supported (must be declared and tested).
- Performance stack choice (NumPy vs Numba vs JAX etc.)—defer until correctness harness exists.

---

## 16) Python Backend Milestone State (SR-L2)

### 16.1 Active capability level

The `python_backend` is governed at **`SR-L2`** (`python_backend/capabilities.py:CAPABILITY_LEVEL`). All golden problems in `sr-golden-problems.yaml` are validated against this level. The capability ordering is:

```
SR-L0 < SR-L1 < SR-L2 < SR-L3 < SR-L4 < SR-L5
```

Unknown levels raise `BackendOptionError(SR-ERR-OPT-001, ...)`.

### 16.2 Operator allowlists at SR-L2

| category | allowed operator IDs |
|---|---|
| binary | `sr.arith.add_v1`, `sr.arith.sub_v1`, `sr.arith.mul_v1`, `sr.math.protected_div_v1` |
| unary | `sr.math.sin_v1`, `sr.math.cos_v1`, `sr.math.abs_v1`, `sr.math.safe_log_v1` |

Operators not in the allowlist are rejected during `_validate_options()` (via `assert_operators_supported()`).

### 16.3 Numeric policy constant `EPS_DENOM`

Defined in `python_backend/policy.py`:

```python
EPS_DENOM = 1e-8
```

Used as the stabilisation denominator in protected operators:
- `protected_div`: returns 1.0 when `|denominator| < EPS_DENOM`
- `safe_log`: computes `log(|x| + EPS_DENOM)` instead of `log(x)`

The constant is included in the policy dict and thus covered by `policy_digest`:

```json
{"eps_denom": "1e-08", ...}
```

### 16.4 Seeded initialisation for `safe_log`

When `sr.math.safe_log_v1` is among the unary operators, the backend injects deterministic seeded expressions into the initial population (see `search.py:generate_seeded_for_safe_log`). These include:

- `safe_log(x0)`
- `abs(x0)`
- `safe_log(abs(x0))`
- `safe_log(x0 + c)` for `c ∈ {0.5, 1.0, 2.0}`
- `safe_log(abs(x0) + c)` for `c ∈ {0.5, 1.0, 2.0}`

This seeding is deterministic given the seed and is subject to `maxsize` constraints.

