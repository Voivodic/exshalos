"""
This module compute quantities from data of simulations.
"""

from typing import Dict, Optional, Union
import numpy as np


# Compute the density grid
def Compute_Density_Grid(
    pos: np.ndarray,
    vel: Optional[np.ndarray] = None,
    mass: Optional[np.ndarray] = None,
    types: Optional[np.ndarray] = None,
    nd: int = 256,
    L: float = 1000.0,
    Om0: float = 0.31,
    z: float = 0.0,
    direction: Optional[int] = None,
    window: Union[str, int] = "CIC",
    R: float = 4.0,
    R_times: float = 5.0,
    interlacing: bool = False,
    folds: int = 1,
    verbose: bool = False,
    nthreads: int = 1,
) -> np.ndarray:
    """
    Compute the density grid of a set of tracers.

    :param pos: Position of the tracers.
    :type pos: numpy.ndarray
    :param vel: Velocities of the particles in the given direction. Fiducial value: None
    :type vel: Optional[numpy.ndarray]
    :param mass: Weight of each tracer. Fiducial value: None
    :type mass: Optional[numpy.ndarray]
    :param types: Type of each tracer. Fiducial value: None
    :type types: Optional[numpy.ndarray]
    :param nd: Number of cells per dimension. Fiducial value: 256
    :type nd: int
    :param L: Size of the box in Mpc/h. Fiducial value: 1000.0
    :type L: float
    :param Om0: Value of the matter overdensity today. Fiducial value: 0.31
    :type Om0: float
    :param z: Redshift of the density grid. Fiducial value: 0.0
    :type z: float
    :param direction: Direction to use to put the particles in redshift space. Fiducial value: None
    :type direction: Optional[int]
    :param window: Density assignment method used to construct the density grid. Fiducial value: "CIC" (Options: "NGP" = 0, "CIC" = 1, "SPHERICAL" = 2, "EXPONENTIAL" = 3)
    :type window: Union[str, int]
    :param R: Smoothing length used in the spherical and exponential windows. Fiducial value: 4.0
    :type R: float
    :param R_times: Scale considered to account for particles (used in the exponential window) in units of R. Fiducial value: 5.0
    :type R_times: float
    :param interlacing: Whether to use or do not use the interlacing technique. Fiducial value: False
    :type interlacing: bool
    :param folds: Number of times that the box will be folded in each direction. Fiducial value: 1
    :type folds: int
    :param verbose: Whether to output information in the C code. Fiducial value: False
    :type verbose: bool
    :param nthreads: Number of threads used by OpenMP. Fiducial value: 1
    :type nthreads: int

    :return: The density grid for each type of tracer and for the grids with and without interlacing (if interlacing is True).
    :rtype: numpy.ndarray
    """
    # Check the precision and convert the arrays
    from .lib.spectrum import check_precision

    nmass = 0
    precision = check_precision()
    if precision == 4:
        pos = pos.astype("float32")
        if vel is not None:
            vel = vel.astype("float32")
        if mass is not None:
            mass = mass.astype("float32")
            nmass = 1
        if vel is not None:
            vel = vel.astype("float32")
        L = np.float32(L)
        R = np.float32(R)
        R_times = np.float32(R_times)
        Om0 = np.float32(Om0)
        z = np.float32(z)
    else:
        pos = pos.astype("float64")
        if vel is not None:
            vel = vel.astype("float32")
        if mass is not None:
            mass = mass.astype("float64")
            nmass = 1
        if vel is not None:
            vel = vel.astype("float64")
        L = np.float64(L)
        R = np.float64(R)
        R_times = np.float64(R_times)
        Om0 = np.float64(Om0)
        z = np.float64(z)

    # Check some inputs
    if types is not None:
        types = np.fabs(types)
        types = types - np.min(types)
        types = types.astype("int32")
        ntypes = len(np.unique(types))
    else:
        ntypes = 1

    if direction == None or direction == -1:
        direction = -1
    elif direction == "x" or direction == "X" or direction == 0:
        direction = 0
    elif direction == "y" or direction == "Y" or direction == 1:
        direction = 1
    elif direction == "z" or direction == "Z" or direction == 2:
        direction = 2
    else:
        raise ValueError("Direction must be None, x, y or z!")

    if window == "NO" or window == "no" or window == 0:
        print(
            "You need to choose some density assigment method to construct the density grid!"
        )
        return None
    elif window == "NGP" or window == "ngp" or window == 1:
        window = 1
    elif window == "CIC" or window == "cic" or window == 2:
        window = 2
    elif window == "SPHERICAL" or window == "spherical" or window == 3:
        window = 3
    elif window == "EXPONENTIAL" or window == "exponential" or window == 4:
        window = 4

    # Call the c function to compute the density grid
    from .lib.spectrum import grid_compute

    grid = grid_compute(
        pos,
        vel,
        mass,
        np.int32(nmass),
        types,
        np.int32(ntypes),
        np.int32(nd),
        L,
        Om0,
        z,
        np.int32(direction),
        np.int32(window),
        R,
        R_times,
        np.int32(interlacing),
        np.int32(folds),
        np.int32(verbose),
        np.int32(nthreads),
    )

    # Reshape the grid to the desired shape
    if interlacing == False and ntypes == 1:
        grid = grid.reshape([nd, nd, nd])
    elif interlacing == False:
        grid = grid.reshape([ntypes, nd, nd, nd])
    elif ntypes == 1:
        grid = grid.reshape([2, nd, nd, nd])

    return grid


# Compute the power spectrum given the density grid
def Compute_Power_Spectrum(
    grid: np.ndarray,
    L: float = 1000.0,
    window: Union[str, int] = "CIC",
    R: float = 4.0,
    Nk: int = 25,
    k_min: Optional[float] = None,
    k_max: Optional[float] = None,
    l_max: int = 0,
    direction: Optional[Union[str, int]] = None,
    folds: int = 1,
    verbose: bool = False,
    nthreads: int = 1,
    ntypes: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Compute the power spectrum from a density grid.

    :param grid: Density grid for all tracers.
    :type grid: numpy.ndarray
    :param L: Size of the box in Mpc/h. Fiducial value: 1000.0
    :type L: float
    :param window: Density assignment method used to construct the density grid. Fiducial value: "CIC" (Options: "NGP" = 0, "CIC" = 1, "SPHERICAL" = 2, "EXPONENTIAL" = 3)
    :type window: Union[str, int]
    :param R: Smoothing length used in the spherical and exponential windows. Fiducial value: 4.0
    :type R: float
    :param Nk: Number of bins in k to compute the power spectra. Fiducial value: 25
    :type Nk: int
    :param k_min: Minimum value of k to compute the power spectra. Fiducial value: None
    :type k_min: Optional[float]
    :param k_max: Maximum value of k to compute the power spectra. Fiducial value: None
    :type k_max: Optional[float]
    :param l_max: Maximum multipole computed. Fiducial value: 0
    :type l_max: int
    :param direction: Direction to use to put the particles in redshift space. Fiducial value: None
    :type direction: Optional[Union[str, int]]
    :param folds: Number of times that the box was folded in each direction. Fiducial value: 1
    :type folds: int
    :param verbose: Whether to output information in the C code. Fiducial value: False
    :type verbose: bool
    :param nthreads: Number of threads used by OpenMP. Fiducial value: 1
    :type nthreads: int
    :param ntypes: Number of different types of tracers. Fiducial value: 1
    :type ntypes: int

    :return: Dictionary with the following keys:

             - "k": Ndarray with the mean wavenumbers
             - "Pk": Ndarray with all possible power spectra
             - "Nk": Ndarray with the number of independent modes per bin

    :rtype: dict
    """
    # Compute some parameters
    if len(grid.shape) == 3:
        interlacing = 0
        ntypes = 1
    elif len(grid.shape) == 5:
        interlacing = 1
        ntypes = grid.shape[0]
    elif len(grid.shape) == 4 and ntypes == 1:
        interlacing = 1
    elif len(grid.shape) == 4 and ntypes > 1:
        interlacing = 0
    nd = grid.shape[-1]
    L = L / folds

    if k_min is None:
        k_min = 2.0 * np.pi / L

    if k_max is None:
        k_max = np.pi / L * nd

    # Check the precision and convert the arrays
    from .lib.spectrum import check_precision

    precision = check_precision()
    if precision == 4:
        grid = grid.astype("float32")
        L = np.float32(L)
        R = np.float32(R)
        k_min = np.float32(k_min)
        k_max = np.float32(k_max)
    else:
        grid = grid.astype("float64")
        L = np.float64(L)
        R = np.float64(R)
        k_min = np.float64(k_min)
        k_max = np.float64(k_max)

    # Set the window function to be de-convolved
    if window == "NO" or window == "no" or window == "No" or window == 0:
        window = 0
    elif window == "NGP" or window == "ngp" or window == 1:
        window = 1
    elif window == "CIC" or window == "cic" or window == 2:
        window = 2
    elif window == "SPHERICAL" or window == "spherical" or window == 3:
        window = 3
    elif window == "EXPONENTIAL" or window == "exponential" or window == 4:
        window = 4

    if direction == None or direction == -1:
        direction = -1
    elif direction == "x" or direction == "X" or direction == 0:
        direction = 0
    elif direction == "y" or direction == "Y" or direction == 1:
        direction = 1
    elif direction == "z" or direction == "Z" or direction == 2:
        direction = 2
    else:
        raise ValueError("Direction must be None, x, y or z!")

    # Call the c function to compute the power spectrum
    from .lib.spectrum import power_compute

    x = power_compute(
        grid,
        np.int32(ntypes),
        np.int32(nd),
        L,
        np.int32(window),
        R,
        np.int32(interlacing),
        np.int32(Nk),
        k_min,
        k_max,
        np.int32(l_max),
        np.int32(direction),
        np.int32(verbose),
        np.int32(nthreads),
    )

    # Reshape the output to the desired shape
    if ntypes == 1 and l_max == 0:
        x["Pk"] = x["Pk"].reshape([Nk])
    elif ntypes == 1:
        x["Pk"] = x["Pk"].reshape([int(l_max / 2) + 1, Nk])
    elif l_max == 0:
        x["Pk"] = x["Pk"].reshape([int(ntypes * (ntypes + 1) / 2), Nk])

    return x


# Compute the power spectrum given the density grid
def Compute_Power_Spectrum_individual(
    grid: np.ndarray,
    pos: np.ndarray,
    L: float = 1000.0,
    window: Union[str, int] = "CIC",
    R: float = 4.0,
    Nk: int = 25,
    k_min: Optional[float] = None,
    k_max: Optional[float] = None,
    l_max: int = 0,
    direction: Optional[Union[str, int]] = None,
    folds: int = 1,
    verbose: bool = False,
    nthreads: int = 1,
    ntypes: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Compute the power spectrum for individual tracers from a density grid.

    :param grid: Density grid for all tracers.
    :type grid: numpy.ndarray
    :param pos: Position of the tracers.
    :type pos: numpy.ndarray
    :param L: Size of the box in Mpc/h. Fiducial value: 1000.0
    :type L: float
    :param window: Density assignment method used to construct the density grid. Fiducial value: "CIC" (Options: "NGP" = 0, "CIC" = 1, "SPHERICAL" = 2, "EXPONENTIAL" = 3)
    :type window: Union[str, int]
    :param R: Smoothing length used in the spherical and exponential windows. Fiducial value: 4.0
    :type R: float
    :param Nk: Number of bins in k to compute the power spectra. Fiducial value: 25
    :type Nk: int
    :param k_min: Minimum value of k to compute the power spectra. Fiducial value: None
    :type k_min: Optional[float]
    :param k_max: Maximum value of k to compute the power spectra. Fiducial value: None
    :type k_max: Optional[float]
    :param l_max: Maximum multipole computed. Fiducial value: 0
    :type l_max: int
    :param direction: Direction to use to put the particles in redshift space. Fiducial value: None
    :type direction: Optional[Union[str, int]]
    :param folds: Number of times that the box will be folded in each direction. Fiducial value: 1
    :type folds: int
    :param verbose: Whether to output information in the C code. Fiducial value: False
    :type verbose: bool
    :param nthreads: Number of threads used by OpenMP. Fiducial value: 1
    :type nthreads: int
    :param ntypes: Number of different types of tracers. Fiducial value: 1
    :type ntypes: int

    :return: Dictionary with the following keys:

             - "k": Ndarray with the mean wavenumbers
             - "Pk": Ndarray with all possible power spectra
             - "Nk": Ndarray with the number of independent modes per bin

    :rtype: dict
    """
    # Compute some parameters
    if len(grid.shape) == 3:
        interlacing = 0
        ntypes = 1
    elif len(grid.shape) == 5:
        interlacing = 1
        ntypes = grid.shape[0]
    elif len(grid.shape) == 4 and ntypes == 1:
        interlacing = 1
    elif len(grid.shape) == 4 and ntypes > 1:
        interlacing = 0
    nd = grid.shape[-1]
    Np = pos.shape[0]
    L = L / folds

    if k_min is None:
        k_min = 2.0 * np.pi / L

    if k_max is None:
        k_max = np.pi / L * nd

    # Check the precision and convert the arrays
    from .lib.spectrum import check_precision

    precision = check_precision()
    if precision == 4:
        grid = grid.astype("float32")
        pos = pos.astype("float32")
        L = np.float32(L)
        R = np.float32(R)
        k_min = np.float32(k_min)
        k_max = np.float32(k_max)
    else:
        grid = grid.astype("float64")
        pos = pos.astype("float64")
        L = np.float64(L)
        R = np.float64(R)
        k_min = np.float64(k_min)
        k_max = np.float64(k_max)

    # Set the window function to be de-convolved
    if window == "NO" or window == "no" or window == "No" or window == 0:
        window = 0
    elif window == "NGP" or window == "ngp" or window == 1:
        window = 1
    elif window == "CIC" or window == "cic" or window == 2:
        window = 2
    elif window == "SPHERICAL" or window == "spherical" or window == 3:
        window = 3
    elif window == "EXPONENTIAL" or window == "exponential" or window == 4:
        window = 4

    if direction == None or direction == -1:
        direction = -1
    elif direction == "x" or direction == "X" or direction == 0:
        direction = 0
    elif direction == "y" or direction == "Y" or direction == 1:
        direction = 1
    elif direction == "z" or direction == "Z" or direction == 2:
        direction = 2
    else:
        raise ValueError("Direction must be None, x, y or z!")

    # Call the c function to compute the power spectrum
    from .lib.spectrum import power_compute_individual

    x = power_compute_individual(
        grid,
        pos,
        np.int32(ntypes),
        np.int32(nd),
        L,
        np.int32(window),
        R,
        np.int32(interlacing),
        np.int32(Nk),
        k_min,
        k_max,
        np.int32(l_max),
        np.int32(direction),
        np.int32(verbose),
        np.int32(nthreads),
    )

    # Reshape the power spectrum
    if ntypes == 1 and l_max == 0:
        x["Pk"] = x["Pk"].reshape([Np, Nk])
    elif ntypes == 1:
        x["Pk"] = x["Pk"].reshape([Np, int(l_max / 2) + 1, Nk])
    elif l_max == 0:
        x["Pk"] = x["Pk"].reshape([Np, ntypes, Nk])

    return x


# Compute the bispectrum given the density grid
def Compute_BiSpectrum(
    grid: np.ndarray,
    L: float = 1000.0,
    window: Union[str, int] = "CIC",
    R: float = 4.0,
    Nk: int = 25,
    k_min: Optional[float] = None,
    k_max: Optional[float] = None,
    folds: int = 1,
    verbose: bool = False,
    nthreads: int = 1,
    ntypes: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Compute the bispectrum from a density grid.

    :param grid: Density grid for all tracers.
    :type grid: numpy.ndarray
    :param L: Size of the box in Mpc/h. Fiducial value: 1000.0
    :type L: float
    :param window: Density assignment method used to construct the density grid. Fiducial value: "CIC" (Options: "NGP" = 0, "CIC" = 1, "SPHERICAL" = 2, "EXPONENTIAL" = 3)
    :type window: Union[str, int]
    :param R: Smoothing length used in the spherical and exponential windows. Fiducial value: 4.0
    :type R: float
    :param Nk: Number of bins in k to compute the tri-spectra. Fiducial value: 25
    :type Nk: int
    :param k_min: Minimum value of k to compute the tri-spectra. Fiducial value: None
    :type k_min: Optional[float]
    :param k_max: Maximum value of k to compute the tri-spectra. Fiducial value: None
    :type k_max: Optional[float]
    :param folds: Number of times that the box was folded in each direction. Fiducial value: 1
    :type folds: int
    :param verbose: Whether to output information in the C code. Fiducial value: False
    :type verbose: bool
    :param nthreads: Number of threads used by OpenMP. Fiducial value: 1
    :type nthreads: int
    :param ntypes: Number of different types of tracers. Fiducial value: 1
    :type ntypes: int

    :return: Dictionary with the following keys:

             - "kP": Ndarray with the mean wavenumbers
             - "Pk": Ndarray with all possible power spectra
             - "Nk": Ndarray with the number of independent modes per bin
             - "kB": Ndarray with the mean tripets of k for the bispectra
             - "Bk": Ndarray with all possible bispectra
             - "Ntri": Number of independent triangles in each bin

    :rtype: dict
    """
    # Compute some parameters
    if len(grid.shape) == 3:
        interlacing = 0
        ntypes = 1
    elif len(grid.shape) == 5:
        interlacing = 1
        ntypes = grid.shape[0]
    elif len(grid.shape) == 4 and ntypes == 1:
        interlacing = 1
    elif len(grid.shape) == 4 and ntypes > 1:
        interlacing = 0
    nd = grid.shape[-1]
    L = L / folds

    if k_min is None:
        k_min = 2.0 * np.pi / L

    if k_max is None:
        k_max = np.pi / L * nd

    # Check the precision and convert the arrays
    from .lib.spectrum import check_precision

    precision = check_precision()
    if precision == 4:
        grid = grid.astype("float32")
        L = np.float32(L)
        R = np.float32(R)
        k_min = np.float32(k_min)
        k_max = np.float32(k_max)
    else:
        grid = grid.astype("float64")
        L = np.float64(L)
        R = np.float64(R)
        k_min = np.float64(k_min)
        k_max = np.float64(k_max)

    if window == "NO" or window == "no" or window == "No" or window == 0:
        window = 0
    elif window == "NGP" or window == "ngp" or window == 1:
        window = 1
    elif window == "CIC" or window == "cic" or window == 2:
        window = 2
    elif window == "SPHERICAL" or window == "spherical" or window == 3:
        window = 3
    elif window == "EXPONENTIAL" or window == "exponential" or window == 4:
        window = 4

    # Call the c function to compute the bispectrum
    from .lib.spectrum import bi_compute

    x = bi_compute(
        grid,
        np.int32(ntypes),
        np.int32(nd),
        L,
        np.int32(window),
        R,
        np.int32(interlacing),
        np.int32(Nk),
        k_min,
        k_max,
        np.int32(verbose),
        np.int32(nthreads),
    )

    # Reshape the bispectrum
    if ntypes == 1:
        x["Pk"] = x["Pk"].reshape([Nk])
        x["Bk"] = x["Bk"].reshape([len(x["Ntri"])])

    return x


# Compute the trispectrum given the density grid
def Compute_TriSpectrum(
    grid: np.ndarray,
    L: float = 1000.0,
    window: Union[str, int] = "CIC",
    R: float = 4.0,
    Nk: int = 25,
    k_min: Optional[float] = None,
    k_max: Optional[float] = None,
    folds: int = 1,
    verbose: bool = False,
    nthreads: int = 1,
    ntypes: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Compute the trispectrum from a density grid.

    :param grid: Density grid for all tracers.
    :type grid: numpy.ndarray
    :param L: Size of the box in Mpc/h. Fiducial value: 1000.0
    :type L: float
    :param window: Density assignment method used to construct the density grid. Fiducial value: "CIC" (Options: "NGP" = 0, "CIC" = 1, "SPHERICAL" = 2, "EXPONENTIAL" = 3)
    :type window: Union[str, int]
    :param R: Smoothing length used in the spherical and exponential windows. Fiducial value: 4.0
    :type R: float
    :param Nk: Number of bins in k to compute the tri-spectra. Fiducial value: 25
    :type Nk: int
    :param k_min: Minimum value of k to compute the tri-spectra. Fiducial value: None
    :type k_min: Optional[float]
    :param k_max: Maximum value of k to compute the tri-spectra. Fiducial value: None
    :type k_max: Optional[float]
    :param folds: Number of times that the box was folded in each direction. Fiducial value: 1
    :type folds: int
    :param verbose: Whether to utput information in the C code. Fiducial value: False
    :type verbose: bool
    :param nthreads: Number of threads used by OpenMP. Fiducial value: 1
    :type nthreads: int
    :param ntypes: Number of different types of tracers. Fiducial value: 1
    :type ntypes: int

    :return: Dictionary with the following keys:

             - "kP": Ndarray with the mean wavenumbers
             - "Pk": Ndarray with all possible power spectra
             - "Nk": Ndarray with the number of independent modes per bin
             - "kT": Ndarray with the mean duplets of k for the trispectra
             - "Tk": Ndarray with all possible trispectra
             - "Tuk": Ndarray with all possible unconnected trispectr
             - "Nsq": Number of independent tetrahedra in each bin

    :rtype: dict
    """
    # Compute some parameters
    if len(grid.shape) == 3:
        interlacing = 0
        ntypes = 1
    elif len(grid.shape) == 5:
        interlacing = 1
        ntypes = grid.shape[0]
    elif len(grid.shape) == 4 and ntypes == 1:
        interlacing = 1
    elif len(grid.shape) == 4 and ntypes > 1:
        interlacing = 0
    nd = grid.shape[-1]
    L = L / folds

    if k_min is None:
        k_min = 2.0 * np.pi / L

    if k_max is None:
        k_max = np.pi / L * nd

    # Check the precision and convert the arrays
    from .lib.spectrum import check_precision

    precision = check_precision()
    if precision == 4:
        grid = grid.astype("float32")
        L = np.float32(L)
        R = np.float32(R)
        k_min = np.float32(k_min)
        k_max = np.float32(k_max)
    else:
        grid = grid.astype("float64")
        L = np.float64(L)
        R = np.float64(R)
        k_min = np.float64(k_min)
        k_max = np.float64(k_max)

    if window == "NO" or window == "no" or window == "No" or window == 0:
        window = 0
    elif window == "NGP" or window == "ngp" or window == 1:
        window = 1
    elif window == "CIC" or window == "cic" or window == 2:
        window = 2
    elif window == "SPHERICAL" or window == "spherical" or window == 3:
        window = 3
    elif window == "EXPONENTIAL" or window == "exponential" or window == 4:
        window = 4

    # Call the c function to compute the trispectrum
    from .lib.spectrum import tri_compute

    x = tri_compute(
        grid,
        np.int32(ntypes),
        np.int32(nd),
        L,
        np.int32(window),
        R,
        np.int32(interlacing),
        np.int32(Nk),
        k_min,
        k_max,
        np.int32(verbose),
        np.int32(nthreads),
    )

    # Reshape the trispectrum
    if ntypes == 1:
        x["Pk"] = x["Pk"].reshape([Nk])
        x["Tk"] = x["Tk"].reshape([len(x["Nsq"])])
        x["Tuk"] = x["Tuk"].reshape([len(x["Nsq"])])

    return x


# Measure the abundance of a list of halo masses
def Compute_Abundance(
    Mh: np.ndarray,
    Mmin: Optional[float] = None,
    Mmax: Optional[float] = None,
    Nm: int = 25,
    Lc: float = 2.0,
    nd: int = 256,
    ndx: Optional[int] = None,
    ndy: Optional[int] = None,
    ndz: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute the abundance from an array of masses.

    :param Mh: Mass of each halo.
    :type Mh: numpy.ndarray
    :param Mmin: Minimum mass used to construct the mass bins. Fiducial value: None
    :type Mmin: Optional[float]
    :param Mmax: Maximum mass used to construct the mass bins. Fiducial value: None
    :type Mmax: Optional[float]
    :param Nm: Number of mass bins. Fiducial value: 25
    :type Nm: int
    :param Lc: Size of each cell in Mpc/h. Fiducial value: 2.0
    :type Lc: float
    :param nd: Number of cells in each direction. Fiducial value: 256
    :type nd: int
    :param ndx: Number of cells in the x direction. Fiducial value: None
    :type ndx: Optional[int]
    :param ndy: Number of cells in the y direction. Fiducial value: None
    :type ndy: Optional[int]
    :param ndz: Number of cells in the z direction. Fiducial value: None
    :type ndz: Optional[int]
    :param verbose: Whether to output information in the C code. Fiducial value: False
    :type verbose: bool

    :return: Dictionary with the following keys:

             - "Mh": Ndarray with the mean mass in the bin
             - "dn": Ndarray with differential abundance in the bin
             - "dn_err": Ndarray with the error in the differential abundance in the bin

    :rtype: dict
    """
    # Check the precision and convert the arrays
    from .lib.spectrum import check_precision

    # Set the maximum and minimum masses to -1 if no value was given
    if Mmin is None:
        Mmin = -1.0
    if Mmax is None:
        Mmax = -1.0

    precision = check_precision()
    if precision == 4:
        Mh = Mh.astype("float32")
        Mmin = np.float32(Mmin)
        Mmax = np.float32(Mmax)
        Lc = np.float32(Lc)
    else:
        Mh = Mh.astype("float64")
        Mmin = np.float64(Mmin)
        Mmax = np.float64(Mmax)
        Lc = np.float64(Lc)

    # Set the number of cells in each dimension
    if ndx is None:
        ndx = nd
    if ndy is None:
        ndy = nd
    if ndz is None:
        ndz = nd

    # Call the c function to compute the abundance
    from .lib.spectrum import abundance_compute

    x = abundance_compute(
        Mh,
        Mmin,
        Mmax,
        np.int32(Nm),
        Lc,
        np.int32(ndx),
        np.int32(ndy),
        np.int32(ndz),
        np.int32(verbose),
    )

    return x


# Measure the b_n bias parameters uising Jens' technic
def Compute_Bias_Jens(
    grid: np.ndarray,
    Mh: np.ndarray,
    flag: np.ndarray,
    Mmin: Optional[float] = None,
    Mmax: Optional[float] = None,
    Nm: int = 20,
    dmin: float = -1.0,
    dmax: float = 1.0,
    Nbins: int = 200,
    Lambda: Optional[float] = None,
    Lc: float = 2.0,
    Central: bool = True,
    Normalized: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute the bias of halos using Jens' method.

    :param grid: Linear (initial) density grid.
    :type grid: numpy.ndarray
    :param Mh: Mass of all halos.
    :type Mh: numpy.ndarray
    :param flag: Flag of all grid cells as being part of a trace or not.
    :type flag: numpy.ndarray
    :param Mmin: Minimum halo mass to be considered. Fiducial value: None
    :type Mmin: Optional[float]
    :param Mmax: Maximum halo mass to be considered. Fiducial value: None
    :type Mmax: Optional[float]
    :param Nm: Number of bins in mass. Fiducial value: 20
    :type Nm: int
    :param dmin: Minimum density to be used in the construction of the histograms in delta. Fiducial value: -1.0
    :type dmin: float
    :param dmax: Maximum density to be used in the construction of the histograms in delta. Fiducial value: 1.0
    :type dmax: float
    :param Nbins: Number of bins to be used in the construction of the histograms in delta. Fiducial value: 200
    :type Nbins: int
    :param Lambda: Scale used to smooth the density field. Fiducial value: None
    :type Lambda: Optional[float]
    :param Lc: Size of the cells of the grid. Fiducial value: 2.0
    :type Lc: float
    :param Central: Whether to use only the central particle of the halo. Fiducial value: True
    :type Central: bool
    :param Normalized: Whether to output the normalized PDF. Fiducial value: False
    :type Normalized: bool

    :return: Dictionay with the following keys:

             - "delta": Ndarray
             - "Mh": Ndarray
             - "Unmasked": Ndarray
             - "Masked": Ndarray

    :rtype: dict
    """
    # Smooth the density field
    if Lambda is not None:
        from .utils import Smooth_Fields

        delta = Smooth_Fields(
            np.copy(grid),
            Lc=Lc,
            k_smooth=Lambda,
            Input_k=False,
            Nfields=1,
            verbose=False,
            nthreads=1,
        )
    else:
        delta = np.copy(grid)

    # Put the values used for Mmin and Mmax
    if Mmin is None:
        Mmin = -1.0
    if Mmax is None:
        Mmax = -1.0

    # Check the precision and convert the arrays
    from .lib.spectrum import check_precision

    precision = check_precision()
    if precision == 4:
        delta = np.float32(delta)
        Mh = np.float32(Mh)
        Mmin = np.float32(Mmin)
        Mmax = np.float32(Mmax)
        dmin = np.float32(dmin)
        dmax = np.float32(dmax)
    else:
        delta = np.float64(delta)
        Mh = np.float64(Mh)
        Mmin = np.float64(Mmin)
        Mmax = np.float64(Mmax)
        dmin = np.float64(dmin)
        dmax = np.float64(dmax)

    # Call the c function that compute the histgram
    from .lib.spectrum import histogram_compute

    x = histogram_compute(
        delta,
        Mh,
        np.int64(flag),
        Mmin,
        Mmax,
        np.int32(Nm),
        dmin,
        dmax,
        np.int32(Nbins),
        np.int32(Central),
    )
    x["sigma2"] = np.mean(delta**2)
    del delta

    # Normalize the histograms
    if Normalized == True:
        x["Unmasked"] = x["Unmasked"] / np.sum(x["Unmasked"])
        x["Masked"] = x["Masked"] / np.sum(x["Masked"], axis=1).reshape([Nm, 1])

    return x


# Find halos and voids from a particle distribution using voro++
def halo_void_finder(
    particles: np.ndarray,
    L: float,
    delta_h: Optional[float] = None,
    delta_v: Optional[float] = None,
    nd: Optional[int] = None,
    r_max: float = 20.0,
) -> Dict[str, np.ndarray]:
    """
    Find spherical halos and voids from a 2D or 3D particle distribution.

    The dimensionality is decided from the shape of ``particles``: (N, 2) is
    treated as 2D and (N, 3) as 3D. The local density is estimated from the
    inverse of the Voronoi cell volumes (areas in 2D).

    :param particles: Array of shape (N, 2) or (N, 3) with the positions.
    :type particles: numpy.ndarray
    :param L: Size of the (cubic) box in Mpc/h.
    :type L: float
    :param delta_h: Halo overdensity threshold. None or <= 0 disables halos. Fiducial value: None
    :type delta_h: Optional[float]
    :param delta_v: Void overdensity threshold. None or <= 0 disables voids. Fiducial value: None
    :type delta_v: Optional[float]
    :param nd: Number of container divisions per dimension used by voro++. Fiducial value: None
    :type nd: Optional[int]
    :param r_max: Maximum radius of a halo/void in Mpc/h. Fiducial value: 20.0
    :type r_max: float

    :return: Dictionary with keys "halos" and "voids", each an ndarray of
             shape (M, dim+1) with columns (x, y(, z), radius).
    :rtype: dict
    """
    # Check if both delta_h and delta_v are None
    if delta_h is None and delta_v is None:
        raise ValueError("Both delta_h and delta_v cannot be None or <= 0!")

    # Check the precision and convert the arrays
    from .lib.halovoid import check_precision

    # Get the precision
    precision = check_precision()

    # Validate the particle array and infer the dimensionality from its shape
    if precision == 4:
        particles = np.ascontiguousarray(particles, dtype=np.float32)
    else:
        particles = np.ascontiguousarray(particles, dtype=np.float64)
    if particles.ndim != 2 or particles.shape[1] not in (2, 3):
        raise ValueError(
            f"particles must have shape (N, 2) or (N, 3), got %{str(particles.shape)}"
        )
    dim = particles.shape[1]

    # Set the value of nd if it was not given
    if nd is None:
        if dim == 2:
            nd = particles.shape[0] // 64
        elif dim == 3:
            nd = particles.shape[0] // 512

    # Normalise the thresholds: None or <= 0 means "do not find"
    find_halos = 0 if (delta_h is None or delta_h <= 0.0) else 1
    find_voids = 0 if (delta_v is None or delta_v <= 0.0) else 1
    delta_h = 0.0 if (delta_h is None or delta_h <= 0.0) else delta_h
    delta_v = 0.0 if (delta_v is None or delta_v <= 0.0) else delta_v

    # Set the correct precision to the floats
    if precision == 4:
        delta_h = np.float32(delta_h)
        delta_v = np.float32(delta_v)
        L = np.float32(L)
        r_max = np.float32(r_max)
    else:
        delta_h = np.float64(delta_h)
        delta_v = np.float64(delta_v)
        L = np.float64(L)
        r_max = np.float64(r_max)

    # Call the C/C++ function of the corresponding dimension
    from .lib.finder import find

    out = find(
        particles,
        np.int32(find_halos),
        np.int32(find_voids),
        delta_h,
        delta_v,
        L,
        np.int32(nd),
        r_max,
    )

    # Reshape to (M, dim+1): columns x(,y(,z)) + radius
    out["halos"] = out["halos"].reshape([-1, dim + 1])
    out["voids"] = out["voids"].reshape([-1, dim + 1])
    return {"halos": out["halos"], "voids": out["voids"]}
