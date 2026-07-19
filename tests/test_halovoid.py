"""
Test the halovoid module
"""

import pytest
import numpy as np
import pyexshalos as exs


@pytest.fixture(scope="module")
def params() -> dict:
    """
    Defines the parameters used by the tests.
    """
    return {
        "L": 1000.0,
        "Nd": 20,
        "Np": 1_00_000,
    }


def test_total_cell_volume(params: dict) -> None:
    """
    Test if the total volume is recovered correctly.
    """
    x = params["L"] * np.random.random((params["Np"], 3))
    volume = exs.simulation.total_volume(x, params["Nd"], params["L"])

    assert np.isclose(volume, params["L"] ** 3, rtol=1e-3)
