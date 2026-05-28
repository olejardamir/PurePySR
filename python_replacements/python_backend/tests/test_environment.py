"""Tests for dependency version ranges and supported Python versions."""

from __future__ import annotations

import sys


def test_python_version_supported():
    print(f"Python {sys.version}")
    assert sys.version_info >= (3, 10), (
        f"Python {sys.version_info.major}.{sys.version_info.minor} < 3.10"
    )


def test_numpy_version_compatible():
    import numpy as np

    v = tuple(int(x) for x in np.__version__.split(".")[:2])
    assert v >= (1, 21), f"numpy {np.__version__} < 1.21"
    print(f"numpy {np.__version__}")


def test_scipy_version_compatible():
    try:
        import scipy

        v = tuple(int(x) for x in scipy.__version__.split(".")[:2])
        assert v >= (1, 7), f"scipy {scipy.__version__} < 1.7"
        print(f"scipy {scipy.__version__}")
    except ImportError:
        pass


def test_pandas_version_compatible():
    import pandas as pd

    v = tuple(int(x) for x in pd.__version__.split(".")[:2])
    assert v >= (1, 3), f"pandas {pd.__version__} < 1.3"
    print(f"pandas {pd.__version__}")


def test_sklearn_version_compatible():
    import sklearn

    v = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    assert v >= (1, 0), f"sklearn {sklearn.__version__} < 1.0"
    print(f"sklearn {sklearn.__version__}")


def test_core_dependencies_importable():
    import numpy
    import scipy.optimize
    import pandas
    import json
    import pickle
    import csv
    from pathlib import Path
    import tempfile
    import copy
    import warnings


def test_backend_imports_clean():
    import os
    os.environ["PYSR_BACKEND"] = "python"

    from python_backend.backend import PythonSRBackend
    from python_backend.eval import evaluate
    from python_backend.expr import OpNode, VarNode, ConstNode
    from python_backend.options import BackendOptions
    from python_backend.hof import HallOfFame
    from python_backend.constant_optimization import optimize_constants
    from python_backend.search import mutate
