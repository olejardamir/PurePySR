"""Tests that the repo-local juliacall/ stub provides the minimal API surface
that PySR_custom imports at runtime, without a Julia binary installed.
"""

from __future__ import annotations

import numpy as np


def test_juliacall_shim_imports():
    import juliacall

    assert hasattr(juliacall, "Main")
    assert hasattr(juliacall, "AnyValue")
    assert hasattr(juliacall, "VectorValue")
    assert hasattr(juliacall, "convert")

    jl = juliacall.Main

    assert hasattr(jl, "seval")
    assert callable(jl.seval)

    result = jl.seval("1 + 1")
    assert callable(result)

    sr = jl.seval("using SymbolicRegression")
    assert callable(sr)

    arr_type = jl.Array
    assert hasattr(arr_type, "__getitem__")

    assert callable(jl.Dict)
    assert callable(jl.NamedTuple)
    assert hasattr(jl, "IOBuffer")
    assert hasattr(jl, "VERSION")
    assert hasattr(jl, "Symbol")
    assert hasattr(jl, "Pair")
    assert hasattr(jl, "Serialization")
    assert juliacall.convert(int, 42) == 42


def test_juliacall_shim_anyvalue_type():
    from juliacall import AnyValue

    assert isinstance(AnyValue, type)


def test_juliacall_shim_vectorvalue_type():
    from juliacall import VectorValue

    assert isinstance(VectorValue, type)


def test_juliacall_shim_symbolic_regression_equation_search():
    """equation_search is accessed via jl.SymbolicRegression.equation_search."""
    import juliacall

    from python_backend.options import BackendOptions

    jl = juliacall.Main
    sr = jl.SymbolicRegression
    assert hasattr(sr, "equation_search")
    assert callable(sr.equation_search)

    opts = BackendOptions(
        binary_operators=["+", "-"],
        unary_operators=[],
    )
    X = np.random.randn(20, 2).astype(np.float64)
    y = np.random.randn(20).astype(np.float64)
    result = sr.equation_search(X, y, options=opts)
    # Returns _ResultWrapper which wraps a dict
    assert hasattr(result, "_result")
    assert "hall_of_fame" in result._result
