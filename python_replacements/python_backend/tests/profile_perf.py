#!/usr/bin/env python3
"""Profile each hot path in the Python backend and report timing breakdowns."""
import os, sys, time, copy
os.environ["PYSR_BACKEND"] = "python"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from python_backend.eval import _evaluate_numeric_fast, evaluate, compute_complexity, _compute_depth, check_constraints, compute_loss
from python_backend.expr import Node, VarNode, ConstNode, OpNode
from python_backend.options import BackendOptions
from python_backend.backend import PythonSRBackend
from python_backend.hof import HallOfFame
from python_backend.constant_optimization import optimize_constants
from python_backend.search import mutate, generate_expression
from python_backend.ops import resolve_operator_tokens

rng = np.random.default_rng(42)

binary_ids = ["sr.arith.add_v1", "sr.arith.sub_v1", "sr.arith.mul_v1",
              "sr.math.protected_div_v1"]
unary_ids = ["sr.math.sin_v1", "sr.math.cos_v1", "sr.math.abs_v1",
             "sr.math.safe_log_v1"]
all_bin_tokens = ["+", "-", "*", "/"]
all_un_tokens = ["sin", "cos", "abs", "safe_log"]

deep_tree = OpNode("sr.math.sin_v1", [
    OpNode("sr.math.cos_v1", [
        OpNode("sr.arith.add_v1", [
            OpNode("sr.arith.mul_v1", [VarNode(0), ConstNode(2.0)]),
            VarNode(1),
        ])
    ])
])

def build_medium_tree() -> Node:
    return generate_expression(
        np.random.default_rng(99), binary_ids, unary_ids, n_features=5, max_depth=6
    )

medium_tree = build_medium_tree()

X_large = np.random.randn(100000, 5).astype(np.float64)
X_med = np.random.randn(500, 10).astype(np.float64)
sin_data = np.sin(X_large[:, 0])
y_med = np.random.randn(500)

results: list[tuple[str, int, float, str]] = []

def add_result(name: str, count: int, elapsed: float, unit: str) -> None:
    results.append((name, count, elapsed, unit))

def print_results() -> None:
    print()
    print("=== Performance Profile ===")
    header = f"{'Operation':<47} {'Count':>7} {'Total (s)':>10} {'Per call':>12}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for name, count, total_s, unit in results:
        if unit == "us":
            per_str = f"{total_s / count * 1e6:>8.2f} us"
        else:
            per_str = f"{total_s / count * 1e3:>8.4f} ms"
        print(f"{name:<47} {count:>7} {total_s:>10.4f} {per_str:>12}")

# ---------------------------------------------------------------------------
# 1. Expression evaluation (fast numeric path)
# ---------------------------------------------------------------------------
print("Profiling 1/12: Expression evaluation ... ", end="", flush=True)
count = 1000
start = time.perf_counter()
for _ in range(count):
    _evaluate_numeric_fast(deep_tree, X_large)
elapsed = time.perf_counter() - start
add_result("evaluate (100k x 5, deep)", count, elapsed, "ms")
print(f"done  [{elapsed/count*1e3:.4f} ms/call]")

# ---------------------------------------------------------------------------
# 2. Complexity computation
# ---------------------------------------------------------------------------
print("Profiling 2/12: Complexity computation ... ", end="", flush=True)
count = 10000
start = time.perf_counter()
for _ in range(count):
    compute_complexity(deep_tree)
elapsed = time.perf_counter() - start
add_result("compute_complexity", count, elapsed, "us")
print(f"done  [{elapsed/count*1e6:.2f} us/call]")

# ---------------------------------------------------------------------------
# 3. Depth computation
# ---------------------------------------------------------------------------
print("Profiling 3/12: Depth computation ... ", end="", flush=True)
count = 10000
start = time.perf_counter()
for _ in range(count):
    _compute_depth(deep_tree)
elapsed = time.perf_counter() - start
add_result("_compute_depth", count, elapsed, "us")
print(f"done  [{elapsed/count*1e6:.2f} us/call]")

# ---------------------------------------------------------------------------
# 4. Constraint checking
# ---------------------------------------------------------------------------
print("Profiling 4/12: Constraint checking ... ", end="", flush=True)
constraints = {"sr.arith.add_v1": 2}
count = 10000
start = time.perf_counter()
for _ in range(count):
    check_constraints(deep_tree, maxsize=50, maxdepth=20, constraints=constraints)
elapsed = time.perf_counter() - start
add_result("check_constraints", count, elapsed, "us")
print(f"done  [{elapsed/count*1e6:.2f} us/call]")

# ---------------------------------------------------------------------------
# 5. Mutation generation
# ---------------------------------------------------------------------------
print("Profiling 5/12: Mutation generation ... ", end="", flush=True)
mut_types = [
    "mutate_constant", "mutate_operator", "mutate_feature",
    "swap_operands", "add_node", "insert_node", "delete_node",
]
mut_rng = np.random.default_rng(7)
count = 1000
start = time.perf_counter()
for i in range(count):
    mt = mut_types[i % len(mut_types)]
    mutate(
        medium_tree, mut_rng, binary_ids, unary_ids,
        n_features=5, maxsize=50, mutation_type=mt,
    )
elapsed = time.perf_counter() - start
add_result("mutate (medium tree)", count, elapsed, "ms")
print(f"done  [{elapsed/count*1e3:.4f} ms/call]")

# ---------------------------------------------------------------------------
# 6. Tree copying (deepcopy)
# ---------------------------------------------------------------------------
print("Profiling 6/12: Tree copying ... ", end="", flush=True)
count = 10000
start = time.perf_counter()
for _ in range(count):
    copy.deepcopy(medium_tree)
elapsed = time.perf_counter() - start
add_result("copy.deepcopy (medium tree)", count, elapsed, "us")
print(f"done  [{elapsed/count*1e6:.2f} us/call]")

# ---------------------------------------------------------------------------
# 7. HOF insertion
# ---------------------------------------------------------------------------
print("Profiling 7/12: HOF insertion ... ", end="", flush=True)
hof = HallOfFame(max_size=50)
hof_rng = np.random.default_rng(13)
entries_data: list[tuple[Node, float, int, str]] = []
for i in range(1000):
    tree = generate_expression(
        hof_rng, binary_ids, unary_ids, n_features=5, max_depth=3
    )
    loss = float(hof_rng.uniform(0.01, 10.0))
    cplx = int(hof_rng.integers(1, 15))
    h = tree.structural_hash()
    entries_data.append((tree, loss, cplx, h))

count = len(entries_data)
start = time.perf_counter()
for tree, loss, cplx, h in entries_data:
    hof.consider(tree, loss, cplx, h)
elapsed = time.perf_counter() - start
add_result("HOF consider (1000 inserts)", count, elapsed, "us")
print(f"done  [{elapsed/count*1e6:.2f} us/insert]")

# ---------------------------------------------------------------------------
# 8. HOF Pareto frontier
# ---------------------------------------------------------------------------
print("Profiling 8/12: HOF Pareto frontier ... ", end="", flush=True)
count = 1000
start = time.perf_counter()
for _ in range(count):
    hof.calculate_pareto_frontier()
elapsed = time.perf_counter() - start
add_result("HOF calculate_pareto_frontier", count, elapsed, "us")
print(f"done  [{elapsed/count*1e6:.2f} us/call]")

# ---------------------------------------------------------------------------
# 9. Constant optimization
# ---------------------------------------------------------------------------
print("Profiling 9/12: Constant optimization ... ", end="", flush=True)
const_tree = OpNode("sr.arith.add_v1", [
    OpNode("sr.math.sin_v1", [
        OpNode("sr.arith.mul_v1", [ConstNode(1.5), VarNode(0)])
    ]),
    ConstNode(0.5),
])
X_const = np.random.randn(10000, 5).astype(np.float64)
y_const = np.sin(X_const[:, 0] * 0.8) + 0.3
count = 100
start = time.perf_counter()
for _ in range(count):
    t = copy.deepcopy(const_tree)
    optimize_constants(
        t, X_const, y_const,
        maxsize=50, maxdepth=20,
        n_iterations=4,
        nrestarts=1,
    )
elapsed = time.perf_counter() - start
add_result("optimize_constants", count, elapsed, "ms")
print(f"done  [{elapsed/count*1e3:.4f} ms/call]")

# ---------------------------------------------------------------------------
# 10. Full backend search (various sizes)
# ---------------------------------------------------------------------------
print("Profiling 10/12: Full backend search ...")

backend = PythonSRBackend()

# 100x5, 2 iter, pop=15, 4bin+3un
X_small = np.random.randn(100, 5).astype(np.float64)
y_small = np.sin(X_small[:, 0]) + 0.1 * np.random.randn(100)

opts_small = BackendOptions(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "abs"],
    niterations=2,
    population_size=15,
    maxsize=20,
    maxdepth=10,
    search_algorithm="baseline",
    ncycles_per_iteration=20,
    parsimony=0.01,
)

start = time.perf_counter()
result_small = backend.equation_search(X_small, y_small, options=opts_small, seed=42)
elapsed_small = time.perf_counter() - start
add_result("search (100x5, 2 iter, pop=15)", 1, elapsed_small, "ms")
print(f"  100x5, 2 iter, pop=15: {elapsed_small:.4f} s")

# 500x10, 2 iter, pop=15, 4bin+3un
X_med2 = np.random.randn(500, 10).astype(np.float64)
y_med2 = np.sin(X_med2[:, 0]) + 0.1 * np.random.randn(500)

opts_med = BackendOptions(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "abs"],
    niterations=2,
    population_size=15,
    maxsize=20,
    maxdepth=10,
    search_algorithm="baseline",
    ncycles_per_iteration=20,
    parsimony=0.01,
)

start = time.perf_counter()
result_med = backend.equation_search(X_med2, y_med2, options=opts_med, seed=42)
elapsed_med = time.perf_counter() - start
add_result("search (500x10, 2 iter, pop=15)", 1, elapsed_med, "ms")
print(f"  500x10, 2 iter, pop=15: {elapsed_med:.4f} s")

# ---------------------------------------------------------------------------
# 11. Batching comparison (500x10)
# ---------------------------------------------------------------------------
print("Profiling 11/12: Batching comparison ...")

X_batch = np.random.randn(500, 10).astype(np.float64)
y_batch = np.sin(X_batch[:, 0]) + 0.1 * np.random.randn(500)

opts_no_batch = BackendOptions(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "abs"],
    niterations=2,
    population_size=15,
    maxsize=20,
    maxdepth=10,
    search_algorithm="baseline",
    ncycles_per_iteration=20,
    parsimony=0.01,
    batching=False,
)

opts_batch = BackendOptions(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "abs"],
    niterations=2,
    population_size=15,
    maxsize=20,
    maxdepth=10,
    search_algorithm="baseline",
    ncycles_per_iteration=20,
    parsimony=0.01,
    batching=True,
    batch_size=32,
)

start = time.perf_counter()
backend.equation_search(X_batch, y_batch, options=opts_no_batch, seed=42)
elapsed_no_batch = time.perf_counter() - start
add_result("search batching=False", 1, elapsed_no_batch, "ms")
print(f"  batching=False: {elapsed_no_batch:.4f} s")

start = time.perf_counter()
backend.equation_search(X_batch, y_batch, options=opts_batch, seed=42)
elapsed_batch = time.perf_counter() - start
add_result("search batching=True (bs=32)", 1, elapsed_batch, "ms")
print(f"  batching=True (bs=32): {elapsed_batch:.4f} s")

# ---------------------------------------------------------------------------
# 13. Multi-output search overhead
# ---------------------------------------------------------------------------
print("Profiling 13/15: Multi-output search ...", end="", flush=True)
X_mo = np.random.randn(200, 5).astype(np.float64)
y_mo = np.column_stack([
    np.sin(X_mo[:, 0]) + 0.1 * np.random.randn(200),
    np.cos(X_mo[:, 1]) + 0.1 * np.random.randn(200),
]).astype(np.float64)
opts_mo = BackendOptions(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos"],
    niterations=2,
    population_size=10,
    maxsize=15,
    maxdepth=8,
    search_algorithm="regularized_evolution",
    ncycles_per_iteration=15,
    parsimony=0.01,
)
start = time.perf_counter()
result_mo = backend.equation_search(X_mo, y_mo, options=opts_mo, seed=42)
elapsed_mo = time.perf_counter() - start
add_result("multi-output (2 targets)", 1, elapsed_mo, "ms")
print(f"  done  [{elapsed_mo:.4f} s]")

# ---------------------------------------------------------------------------
# 14. Artifact writing profile (simulate HOF/CSV export)
# ---------------------------------------------------------------------------
print("Profiling 14/15: Artifact writing ...", end="", flush=True)
import json, tempfile, pathlib
hof_large = HallOfFame(max_size=200)
wrng = np.random.default_rng(42)
for i in range(200):
    t = generate_expression(wrng, binary_ids, unary_ids, n_features=5, max_depth=4)
    loss = float(wrng.uniform(0.001, 5.0))
    cplx = int(wrng.integers(1, 30))
    h = t.structural_hash()
    hof_large.consider(t, loss, cplx, h)

entries = hof_large.entries()
count_writes = 1000
tmpdir = pathlib.Path(tempfile.mkdtemp())
start = time.perf_counter()
for _ in range(count_writes):
    p = tmpdir / "hof.json"
    with open(p, "w") as f:
        json.dump(entries, f, default=str)
    p.unlink()
elapsed_write = time.perf_counter() - start
add_result("json dump HOF (200 entries)", count_writes, elapsed_write, "us")
print(f"  done  [{elapsed_write/count_writes*1e6:.2f} us/call]")

# CSV writing
import csv
start = time.perf_counter()
for _ in range(count_writes):
    p = tmpdir / "hof.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Complexity", "Loss", "Equation"])
        for e in entries:
            w.writerow([e["complexity"], e["loss"], e["canonical_expression"]])
    p.unlink()
elapsed_csv = time.perf_counter() - start
add_result("csv write HOF (200 entries)", count_writes, elapsed_csv, "us")
print(f"  done  [{elapsed_csv/count_writes*1e6:.2f} us/call]")

# ---------------------------------------------------------------------------
# 15. Constant optimization restart strategy (nrestarts=1 vs 3)
# ---------------------------------------------------------------------------
print("Profiling 15/15: Constant opt restart strategy ...")
const_tree2 = OpNode("sr.arith.add_v1", [
    OpNode("sr.math.sin_v1", [
        OpNode("sr.arith.mul_v1", [ConstNode(1.5), VarNode(0)])
    ]),
    ConstNode(0.5),
])
X_co = np.random.randn(5000, 5).astype(np.float64)
y_co = np.sin(X_co[:, 0] * 0.8) + 0.3
count_co = 50
start = time.perf_counter()
for _ in range(count_co):
    t = copy.deepcopy(const_tree2)
    optimize_constants(t, X_co, y_co, maxsize=50, maxdepth=20, n_iterations=4, nrestarts=1)
elapsed_r1 = time.perf_counter() - start
add_result("optimize_constants (nrestarts=1)", count_co, elapsed_r1, "ms")
print(f"  nrestarts=1: {elapsed_r1/count_co*1e3:.4f} ms/call")

start = time.perf_counter()
for _ in range(count_co):
    t = copy.deepcopy(const_tree2)
    optimize_constants(t, X_co, y_co, maxsize=50, maxdepth=20, n_iterations=4, nrestarts=3)
elapsed_r3 = time.perf_counter() - start
add_result("optimize_constants (nrestarts=3)", count_co, elapsed_r3, "ms")
print(f"  nrestarts=3: {elapsed_r3/count_co*1e3:.4f} ms/call")

# ---------------------------------------------------------------------------
# Print results table
# ---------------------------------------------------------------------------
print_results()

opts_single = BackendOptions(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "abs"],
    niterations=2,
    population_size=15,
    maxsize=20,
    maxdepth=10,
    search_algorithm="regularized_evolution",
    ncycles_per_iteration=20,
    parsimony=0.01,
    populations=1,
    migration=True,
    topn=20,
)

opts_multi = BackendOptions(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "abs"],
    niterations=2,
    population_size=15,
    maxsize=20,
    maxdepth=10,
    search_algorithm="regularized_evolution",
    ncycles_per_iteration=20,
    parsimony=0.01,
    populations=2,
    migration=True,
    topn=20,
)

X_pop = np.random.randn(300, 5).astype(np.float64)
y_pop = np.sin(X_pop[:, 0]) + 0.1 * np.random.randn(300)

start = time.perf_counter()
backend.equation_search(X_pop, y_pop, options=opts_single, seed=42)
elapsed_single = time.perf_counter() - start
add_result("search populations=1", 1, elapsed_single, "ms")
print(f"  populations=1: {elapsed_single:.4f} s")

start = time.perf_counter()
backend.equation_search(X_pop, y_pop, options=opts_multi, seed=42)
elapsed_multi = time.perf_counter() - start
add_result("search populations=2", 1, elapsed_multi, "ms")
print(f"  populations=2: {elapsed_multi:.4f} s")

# ---------------------------------------------------------------------------
# Print results table
# ---------------------------------------------------------------------------
print_results()
