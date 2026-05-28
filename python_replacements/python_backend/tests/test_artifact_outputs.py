"""Validate that PySRRegressor.fit produces the expected output artifacts.

The Python backend via PySRRegressor.fit() produces:
  - checkpoint.pkl in output_directory_ / run_id_
  - model.equation_file_contents_ (pandas DataFrame of results)
  - model.output_directory_ and model.run_id_ are populated
"""

from __future__ import annotations

import os
import pickle

import numpy as np

os.environ["PYSR_BACKEND"] = "python"


def _run_tiny_fit():
    """Return (model, artifact_dir) after a minimal fit."""
    from pysr import PySRRegressor

    X = np.random.randn(30, 2).astype(np.float64)
    y = (X[:, 0] ** 2).astype(np.float64)
    model = PySRRegressor(
        niterations=2,
        population_size=10,
        tournament_selection_n=5,
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        maxsize=10,
        verbosity=0,
        progress=False,
    )
    model.fit(X, y)
    artifact_dir = os.path.join(model.output_directory_, model.run_id_)
    return model, artifact_dir


def test_output_directory_populated():
    """model.output_directory_ and model.run_id_ are set after fit."""
    model, artifact_dir = _run_tiny_fit()
    assert model.output_directory_ is not None
    assert model.run_id_ is not None
    assert os.path.isdir(artifact_dir), f"artifact dir {artifact_dir} not found"


def test_checkpoint_pkl_produced():
    """A checkpoint.pkl with the PySRRegressor model is written."""
    _, artifact_dir = _run_tiny_fit()
    path = os.path.join(artifact_dir, "checkpoint.pkl")
    assert os.path.isfile(path), f"checkpoint.pkl not found in {artifact_dir}"
    with open(path, "rb") as f:
        data = pickle.load(f)
    # The checkpoint contains the full PySRRegressor model
    from pysr import PySRRegressor
    assert isinstance(data, PySRRegressor), (
        f"expected PySRRegressor, got {type(data)}"
    )
    import pandas as pd
    assert isinstance(data.equations_, pd.DataFrame)
    assert len(data.equations_) > 0


def test_equation_file_contents_populated():
    """model.equation_file_contents_ is a non-empty list of DataFrames."""
    import pandas as pd
    model, _ = _run_tiny_fit()
    assert hasattr(model, "equation_file_contents_")
    contents = model.equation_file_contents_
    assert isinstance(contents, list), f"expected list, got {type(contents)}"
    assert len(contents) > 0
    df = contents[0]
    assert isinstance(df, pd.DataFrame), f"expected DataFrame, got {type(df)}"
    assert "equation" in df.columns
