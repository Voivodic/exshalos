"""
Test the mock module functions using pytest.
"""

# Import the core modules
from typing import Any

import numpy as np
import pytest

# Import the module with the mock functions
from pyexshalos import mock


# --- Fixtures for test setup ---


@pytest.fixture(scope="module")
def sample_power_spectrum() -> tuple[np.ndarray, np.ndarray]:
    """Create a simple power spectrum for testing."""
    k = np.logspace(-3, 1, 50)
    pk = 1e4 * k**-1.5
    return k, pk


@pytest.fixture(scope="module")
def sample_halo_catalogue() -> tuple[np.ndarray, np.ndarray]:
    """Create a simple halo catalogue for testing."""
    # Create a small halo catalogue with 10 halos
    pos_h = np.random.random((10, 3)) * 512.0  # Positions in a 512 Mpc/h box
    m_h = np.logspace(12, 14, 10)  # Masses between 1e12 and 1e14 Msun/h
    return pos_h, m_h


# --- Test Functions ---


def test_generate_halos_box_from_pk(sample_power_spectrum: tuple[np.ndarray, np.ndarray]) -> None:
    """Test the generate_halos_box_from_pk function."""
    k, pk = sample_power_spectrum
    
    # Run with small grid size for fast testing
    result = mock.generate_halos_box_from_pk(
        k=k,
        pk=pk,
        nd=8,  # Small grid for testing
        cell_size=8.0,
        seed=42,
        verbose=False,
    )
    
    # Check that the result is a dictionary with expected keys
    assert isinstance(result, dict)
    assert "posh" in result
    assert "Mh" in result
    
    # Check that positions and masses are reasonable
    assert result["posh"].shape[1] == 3  # 3D positions
    assert result["Mh"].shape[0] == result["posh"].shape[0]  # Same number of halos
    assert np.all(result["Mh"] > 0)  # Positive masses


def test_generate_halos_box_from_grid(sample_power_spectrum: tuple[np.ndarray, np.ndarray]) -> None:
    """Test the generate_halos_box_from_grid function."""
    k, pk = sample_power_spectrum
    
    # Create a simple density grid
    grid = np.random.normal(0, 1, (8, 8, 8))  # Small grid for testing
    
    result = mock.generate_halos_box_from_grid(
        grid=grid,
        k=k,
        pk=pk,
        cell_size=8.0,
        verbose=False,
    )
    
    # Check that the result is a dictionary with expected keys
    assert isinstance(result, dict)
    assert "posh" in result
    assert "Mh" in result
    
    # Check that positions and masses are reasonable
    assert result["posh"].shape[1] == 3  # 3D positions
    assert result["Mh"].shape[0] == result["posh"].shape[0]  # Same number of halos
    assert np.all(result["Mh"] > 0)  # Positive masses


def test_generate_galaxies_from_halos(sample_halo_catalogue: tuple[np.ndarray, np.ndarray]) -> None:
    """Test the generate_galaxies_from_halos function."""
    pos_h, m_h = sample_halo_catalogue
    
    result = mock.generate_galaxies_from_halos(
        pos_h=pos_h,
        m_h=m_h,
        seed=42,
        verbose=False,
    )
    
    # Check that the result is a dictionary with expected keys
    assert isinstance(result, dict)
    assert "posg" in result
    
    # Check that positions are reasonable
    assert result["posg"].shape[1] == 3  # 3D positions


def test_split_galaxies(sample_halo_catalogue: tuple[np.ndarray, np.ndarray]) -> None:
    """Test the split_galaxies function."""
    pos_h, m_h = sample_halo_catalogue
    
    # First generate galaxies to get flags
    galaxy_result = mock.generate_galaxies_from_halos(
        pos_h=pos_h,
        m_h=m_h,
        seed=42,
        out_flag=True,
        verbose=False,
    )
    
    # Extract flags (1 for centrals, 0 for satellites)
    flag = galaxy_result["flag"].astype(int)
    
    # Split galaxies into types
    galaxy_types = mock.split_galaxies(
        m_h=np.repeat(m_h, 5),  # Repeat halo masses for galaxies
        flag=flag,
        seed=42,
        verbose=False,
    )
    
    # Check that result is an array of integers
    assert isinstance(galaxy_types, np.ndarray)
    assert galaxy_types.dtype == np.integer
    
    # Check that types are either 0 or 1 (two types)
    unique_types = np.unique(galaxy_types)
    assert set(unique_types).issubset({0, 1})
