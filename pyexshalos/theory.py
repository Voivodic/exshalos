"""
This module compute some theoretical quantities.
"""

# Import for annotations
from typing import Any, Optional, cast

# Import numpy
import numpy as np
from numpy.typing import NDArray


# Get the mass of each cell given its size
def get_cell_mass(omega_m0: float = 0.31, cell_size: float = 2.0) -> float:
    """
    Compute the mass of a cell in the density grid.

    :param omega_m0: Omega matter at z=0. Fiducial value: 0.31
    :type omega_m0: float
    :param cell_size: Size of each cell in Mpc/h. Fiducial value: 2.0
    :type cell_size: float

    :return: Mass of the cell in units of solar masses per h.
    :rtype: float
    """
    return 2.775e11 * omega_m0 * np.power(cell_size, 3.0)


# Get the size of each cell given its mass
def get_cell_size(omega_m0: float = 0.31, m_cell: float = 8.5e10) -> float:
    """
    Get the size of each cell given its mass.

    :param omega_m0: Omega matter at z=0. Fiducial value: 0.31
    :type omega_m0: float
    :param m_cell: Mass of the cell in units of solar masses per h.
                   Fiducial value: 8.5e+10
    :type m_cell: float

    :return: Size of the cell in Mpc/h.
    :rtype: float
    """
    return np.power(m_cell / (2.775e11 * omega_m0), 1.0 / 3.0)


# Get Om at any redshift
def get_omz(
    z: NDArray[np.floating], omega_m0: float = 0.31
) -> NDArray[np.floating]:
    """
    Return the value of the matter overdensity at a given redshift.

    :param z: Redshift.
    :type z: NDArray[np.floating]
    :param omega_m0: Omega matter at z=0. Fiducial value: 0.31
    :type omega_m0: float

    :return: Matter overdensity at redshift z.
    :rtype: NDArray[np.floating]
    """
    return (
        omega_m0
        * np.power(1.0 + z, 3.0)
        / (omega_m0 * np.power(1.0 + z, 3.0) + (1.0 - omega_m0))
    )


# Get delta_c (critical matter overdensity for halo formation) at any redshift
def get_deltac(
    z: NDArray[np.floating], omega_m0: float = 0.31
) -> NDArray[np.floating]:
    """
    Return the value of delta_c (matter density contrast for halo
    formation) following a fit.

    :param z: Redshift.
    :type z: NDArray[np.floating]
    :param omega_m0: Omega matter at z=0. Fiducial value: 0.31
    :type omega_m0: float

    :return: Value of delta_c at redshift z.
    :rtype: NDArray[np.floating]
    """
    return 1.686 * np.power(get_omz(z, omega_m0), 0.0055)


# Get H(z) (Hubble constant at z)
def get_hz(
    z: NDArray[np.floating], omega_m0: float = 0.31
) -> NDArray[np.floating]:
    """
    Return the Hubble function, in units of 100*h, at a given redshift.

    :param z: Redshift.
    :type z: NDArray[np.floating]
    :param omega_m0: Omega matter at z=0. Fiducial value: 0.31
    :type omega_m0: float

    :return: Hubble function in units of 100*h at redshift z.
    :rtype: NDArray[np.floating]
    """
    return np.sqrt(omega_m0 * np.power(1.0 + z, 3.0) + (1.0 - omega_m0))


# Get H(a) (Hubble constant at a)
def get_ha(
    a: NDArray[np.floating], omega_m0: float = 0.31
) -> NDArray[np.floating]:
    """
    Return the Hubble function, in units of 100*h, at a given scale factor.

    :param a: Scale factor.
    :type a: NDArray[np.floating]
    :param omega_m0: Omega matter at z=0. Fiducial value: 0.31
    :type omega_m0: float

    :return: Hubble function in units of 100*h at scale factor a.
    :rtype: NDArray[np.floating]
    """
    return np.sqrt(omega_m0 * np.power(a, -3.0) + (1.0 - omega_m0))


# Compute the comoving radial distance at a given redshift
def get_radial_distance(
    z: NDArray[np.floating], omega_m0: float = 0.31
) -> NDArray[np.floating]:
    """
    Return the comoving radial distance, in units of Mpc/h.

    :param z: Redshift.
    :type z: numpy.ndarray
    :param omega_m0: Omega matter at z=0. Fiducial value: 0.31
    :type omega_m0: float

    :return: Distance as a function of redshift.
    :rtype: numpy.ndarray
    """
    # Import cumulative_trapezoid from scipy.integrate
    from scipy.integrate import (
        quad,  # pyright: ignore[reportUnknownVariableType]
    )

    # Compute the integrand
    def _integrand(s: NDArray[np.floating]) -> NDArray[np.floating]:
        return 1.0 / (s**2 * get_ha(s, omega_m0))

    # Compute the integral for all z's
    a = 1.0 / (1.0 + z)
    if len(z.shape) == 0:
        resp = cast(NDArray[np.floating], quad(_integrand, 1.0, a)[0])
    else:
        resp = np.zeros(len(z))
        for i in range(len(z)):
            resp[i] = quad(_integrand, 1.0, a[i])[0]

    # Compute the integral
    return -2997.92 * np.array(resp)


# Compute the angular diameter distance at a given redshift
def get_angular_distance(
    z: NDArray[np.floating],
    ra_0: float,
    ra_1: float,
    dec_0: float,
    dec_1: float,
    omega_m0: float = 0.31,
) -> NDArray[np.floating]:
    """
    Return the comoving angular distance, in units of Mpc/h.

    :param z: Redshift.
    :type z: numpy.ndarray
    :param ra_0: Right ascension of the point 0.
    :type ra_0: float
    :param ra_1: Right ascension of the point 1.
    :type ra_1: float
    :param dec_0: Declination of the point 0.
    :type dec_0: float
    :param dec_1: Declination of the point 1.
    :type dec_1: float
    :param omega_m0: Omega matter at z=0. Fiducial value: 0.31
    :type omega_m0: float

    :return: Angular distance in units of Mpc/h.
    :rtype: numpy.ndarray
    """
    # Compute the angular diameter distance
    r = get_radial_distance(z, omega_m0)
    da = r / (1.0 + z)

    # Compute the angular distance between the objects
    theta = np.arccos(
        np.sin(dec_0) * np.sin(dec_1)
        + np.cos(dec_0) * np.cos(dec_1) * np.cos(ra_0 - ra_1)
    )

    # Compute the angular diameter distance
    da = r / (1.0 + z) * theta

    return da


# Get the shape of a box needed to enclose a volume of the sky
def get_box_shape(
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    z_min: float,
    z_max: float,
    omega_m0: float = 0.31,
    in_rad: bool = True,
) -> tuple[float, float, float]:
    """
    Compute the shape of a box needed to enclose a volume of the sky.

    :param ra_min: Minimum right ascension.
    :type ra_min: float
    :param ra_max: Maximum right ascension.
    :type ra_max: float
    :param dec_min: Minimum declination.
    :type dec_min: float
    :param dec_max: Maximum declination.
    :type dec_max: float
    :param z_min: Minimum redshift.
    :type z_min: float
    :param z_max: Maximum redshift.
    :type z_max: float
    :param omega_m0: Omega matter at z=0. Fiducial value: 0.31
    :type omega_m0: float

    :return: Tuple containing the shape of the box in units of Mpc/h.
    :rtype: tuple[float, float, float]
    """
    # Convert to radians
    if not in_rad:
        ra_min = ra_min * np.pi / 180.0
        ra_max = ra_max * np.pi / 180.0
        dec_min = dec_min * np.pi / 180.0
        dec_max = dec_max * np.pi / 180.0

    # Compute the radial distances
    dz_min: float = get_radial_distance(np.array(z_min), omega_m0).item()
    dz_max: float = get_radial_distance(np.array(z_max), omega_m0).item()

    # Compute the size of the box along each dimension
    lz: float = np.fabs(dz_max - dz_min).item()
    ly: float = (np.max([dz_max, dz_min]) * np.fabs(dec_max - dec_min)).item()
    dec_max_width: float = 0.0
    if (dec_min < 0.0 and dec_max < 0.0) or (dec_min > 0.0 and dec_max > 0.0):
        dec_max_width = np.min([np.fabs(dec_max), np.fabs(dec_min)])
    lx: float = (
        np.max([dz_max, dz_min])
        * np.fabs(ra_max - ra_min)
        * np.cos(dec_max_width)
    ).item()

    return (lx, ly, lz)


# Get dH(a) (derivative of H(a) with respect to a)
def get_dha(
    a: NDArray[np.floating], omega_m0: float = 0.31
) -> NDArray[np.floating]:
    """
    Return the derivative of the Hubble's function, with respect to a in
    units of 100*h, at a given scale factor.

    :param a: Scale factor. Fiducial value: 1.0
    :type a: NDArray[np.floating]
    :param omega_m0: Omega matter at z=0. Fiducial value: 0.31
    :type omega_m0: float

    :return: Derivative of the Hubble's function in units of 100*h at
             scale factor a.
    :rtype: NDArray[np.floating]
    """
    return -1.5 * omega_m0 * np.power(a, -4.0) / get_ha(a, omega_m0)


# Get the growth rate at z
def get_fz(
    z: NDArray[np.floating], omega_m0: float = 0.31
) -> NDArray[np.floating]:
    """
    Return the growth rate at a given redshift.

    :param z: Redshift. Fiducial value: 0.0
    :type z: float
    :param omega_m0: Omega matter at z=0. Fiducial value: 0.31
    :type omega_m0: float

    :return: Growth rate at redshift z.
    :rtype: float
    """
    return np.power(get_omz(z, omega_m0), 0.5454)


# Define the system of differential equations used to compute the growth
# function
def growth_eq(
    y: tuple[NDArray[np.floating], NDArray[np.floating]],
    a: NDArray[np.floating],
    omega_m0: float = 0.31,
) -> NDArray[np.floating]:
    """
    Define the system of differential equations used to compute the growth
    function.

    :param y: Tuple containing the density contrast (d) and its
              derivative (v).
    :type y: tuple of float
    :param a: Scale factor. Fiducial value.
    :type a: float
    :param omega_m0: Omega matter at z=0. Fiducial value: 0.31
    :type omega_m0: float

    :return: Array containing the derivatives of density contrast and
        its velocity.
    :rtype: numpy.ndarray
    """
    d, v = y
    dydt = np.array([
        v,
        -(3.0 / a + get_dha(a, omega_m0) / get_ha(a, omega_m0)) * v
        + 3.0
        / 2.0
        * omega_m0
        / (np.power(get_ha(a, omega_m0), 2.0) * np.power(a, 5.0))
        * d,
    ])
    return dydt


# Return the growth function
def get_dz(
    omega_m0: float = 0.31,
    z_max: float = 1000,
    z_min: float = -0.5,
    n_zs: int = 1000,
) -> dict[str, NDArray[np.floating]]:
    """
    Compute the growth function over a range of redshifts.

    :param omega_m0: Omega matter at z=0. Fiducial value: 0.31
    :type omega_m0: float
    :param z_max: Maximum redshift to consider. Fiducial value: 1000
    :type z_max: float
    :param z_min: Minimum redshift to consider. Fiducial value: -0.5
    :type z_min: float
    :param n_zs: Number of redshift steps. Fiducial value: 1000
    :type n_zs: int

    :return: dictionary with the keys
        - "z": Ndarray with redshifts
        - "a": Ndarray with scale factors
        - "dz": Ndarray with growth factors
        - "d_dz": Ndarray with derivatives of the growth factor

    :rtype: dict
    """
    from scipy.integrate import (
        odeint,  # pyright: ignore[reportUnknownVariableType]
    )

    resp: dict[str, NDArray[np.floating]] = {}

    # Set the initial conditions
    a = cast(
        NDArray[np.floating],
        np.logspace(
            np.log10(1.0 / (z_max + 1.0)), np.log10(1.0 / (z_min + 1.0)), n_zs
        ),
    )
    resp["z"] = 1.0 / a - 1.0
    resp["a"] = a
    d0 = a[0]
    dd0 = 1.0
    y0 = [d0, dd0]

    # Solve the Growth function equation
    sol = cast(
        NDArray[np.floating], odeint(growth_eq, y0, a, args=(omega_m0,))
    )
    resp["dz"] = sol[:, 0]
    resp["d_dz"] = sol[:, 1]

    return resp


# Window (top-hat) function in Fourier space
def wth(k: NDArray[np.floating], r: float) -> NDArray[np.floating]:
    """
    Compute the top-hat window function in Fourier space.

    :param k: Wavenumber.
    :type k: numpy.ndarray
    :param r: Smoothing radius.
    :type r: float

    :return: Window function value.
    :rtype: np.ndarray
    """
    resp = (
        3.0 / (np.power(k * r, 2)) * (np.sin(k * r) / (k * r) - np.cos(k * r))
    )
    return resp


# Compute the variance of the linear density field
def compute_sigma(
    k: NDArray[np.floating],
    pk: NDArray[np.floating],
    r: Optional[NDArray[np.floating]] = None,
    m: Optional[NDArray[np.floating]] = None,
    omega_m0: float = 0.31,
    z: float = 0.0,
) -> NDArray[np.floating]:
    """
    Compute the variance of the density field.

    :param k: Wavenumbers of the power spectrum.
    :type k: numpy.ndarray
    :param pk: Power spectrum.
    :type pk: numpy.ndarray
    :param r: Smoothing radius (used to compute the mass). Fiducial value: None
    :type r: Optional[numpy.ndarray]
    :param m: Mass. Fiducial value: None
    :type m: Optional[numpy.ndarray]
    :param omega_m0: Omega matter at z=0 (used to compute the mass).
                     Fiducial value: 0.31
    :type omega_m0: float
    :param z: Redshift. Fiducial value: 0.0
    :type z: float

    :return: Variance of the density field on the given scales.
    :rtype: numpy.ndarray
    """
    from scipy.integrate import (
        simpson,  # pyright: ignore[reportUnknownVariableType]
    )

    # Compute R(M) if r_smooth is not provided
    if r is None:
        if m is None:
            raise ValueError(
                "You have to provide either the mass array (m) or "
                "the radius array (r_smooth)!"
            )
        else:
            r = np.power(
                3.0
                * m
                / (4.0 * np.pi * 2.775e11 * omega_m0 * np.power(1 + z, 3.0)),
                1.0 / 3.0,
            )
    r = cast(NDArray[np.floating], r)

    # Evaluate sigma
    # At this point, r_smooth is guaranteed to be an NDArray[np.floating]
    # due to the check above
    n_r = len(r)
    sigma = np.zeros(n_r)
    for j in range(n_r):
        kt = k[k <= 2.0 * np.pi / r[j]]
        p_t = pk[k <= 2.0 * np.pi / r[j]]

        sigma[j] = np.sqrt(
            cast(
                NDArray[np.floating],
                simpson(p_t * kt * kt * np.power(wth(kt, r[j]), 2.0), x=kt)
                / (2.0 * np.pi * np.pi),
            )
        )

    return sigma


# Compute the derivative of sigma with respect to M
def dlns_dlnm(
    m: NDArray[np.floating], sigma: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Compute the variance of the logarithm of sigma with respect to the
    logarithm of mass using finite differences.

    :param m: Mass array.
    :type m: numpy.ndarray
    :param sigma: Variance array.
    :type sigma: numpy.ndarray

    :return: Array containing the derivatives of ln(sigma)
             with respect to ln(m).
    :rtype: numpy.ndarray
    """
    result: NDArray[np.floating] = np.zeros(len(m))

    result[0] = (np.log(sigma[1]) - np.log(sigma[0])) / (
        np.log(m[1]) - np.log(m[0])
    )
    result[1:-1] = (np.log(sigma[2:]) - np.log(sigma[:-2])) / (
        np.log(m[2:]) - np.log(m[:-2])
    )
    result[-1] = (np.log(sigma[-1]) - np.log(sigma[-2])) / (
        np.log(m[-1]) - np.log(m[-2])
    )

    return result


# Multiplicity function
def fh(
    s: NDArray[np.floating],
    model: int | str = "PS",
    theta: Optional[float | NDArray[np.floating]] = None,
    delta_c: Optional[float] = None,
    omega_m0: float = 0.31,
    z: float = 0.0,
) -> NDArray[np.floating]:
    """
    Compute the halo mass function (HMF) based on different models.

    :param s: Variance of the density field.
    :type s: numpy.ndarray
    :param model: HMF model to use (0: Press-Schechter, 1: Sheth-Tormen,
                  2: Tinker, 3: Linear Diffusive Barrier).
                  Can also be a string ("PS", "ST", "Tinker", "2LDB").
                  Fiducial value: "PS"
    :type model: Union[int, str]
    :param theta: Model parameters.
                  For Sheth-Tormen: [a, b, p], for Tinker: Delta,
                  for Linear Diffusive Barrier: [b, D, dv, J_max].
    :type theta: Optional[Union[float, np.ndarray]]
    :param delta_c: Critical density for collapse. Fiducial value: None
    :type delta_c: Optional[float]
    :param omega_m0: Omega matter at z=0. Fiducial value: 0.31
    :type omega_m0: float
    :param z: Redshift. Fiducial value: 0.0
    :type z: float

    :return: Array containing the multiplicity function.
    :rtype: numpy.ndarray
    """
    from scipy.special import binom

    # Compute critical density if it was not given
    if delta_c is None:
        delta_c = get_deltac(z=np.array(z), omega_m0=omega_m0).item()
    nu: NDArray[np.floating] = delta_c / s
    result: NDArray[np.floating] = np.zeros(len(s))

    # Press-Schechter
    if model == 0 or model in ["ps", "PS", "1SB"]:
        result = np.sqrt(2.0 / np.pi) * nu * np.exp(-nu * nu / 2)

    # Sheth-Tormen
    elif model == 1 or model in ["ST", "st", "elliptical"]:
        a_st_param: float
        b_st_param: float
        p_st_param: float
        if theta is not None:
            # Expecting theta to be a 1D NDArray of 3 floating-point parameters
            if isinstance(theta, np.ndarray):
                if not theta.shape == (3,):
                    raise ValueError(
                        "Theta for Sheth-Tormen model must be a 1D numpy array"
                        "of 3 parameters [a, b, p]."
                    )
                a_st_param, b_st_param, p_st_param = (
                    float(theta[0]),
                    float(theta[1]),
                    float(theta[2]),
                )
            else:
                raise ValueError(
                    "Theta for Sheth-Tormen model must be a 1D numpy array "
                    "of 3 parameters [a, b, p]."
                )
        else:
            a_st_param, b_st_param, p_st_param = 0.7, 0.4, 0.6

        b_st_val: NDArray[np.floating] = (
            np.sqrt(a_st_param)
            * delta_c
            * (1.0 + b_st_param * np.power(a_st_param * nu * nu, -p_st_param))
        )
        coeff_a: float = 0.0
        for i in range(6):
            coeff_a += np.power(-1, i) * binom(p_st_param, i)

        result = (
            np.sqrt(2.0 * a_st_param / np.pi)
            * nu
            * np.exp(-b_st_val * b_st_val / (2.0 * s * s))
            * (
                1.0
                + b_st_param
                * coeff_a
                * np.power(a_st_param * nu * nu, -p_st_param)
            )
        )

    # Tinker
    elif model == 2 or model in ["Tinker", "tinker", "TINKER"]:
        delta_val: float
        if theta is not None:
            # Expecting theta to be a single scalar float or a 1-element array
            if isinstance(theta, np.ndarray):
                if theta.size != 1:
                    raise ValueError(
                        "Theta for Tinker model must be a single scalar "
                        "or an array of 1 element."
                    )
                delta_val = float(theta.item())
            else:  # theta is float
                delta_val = float(theta)
        else:
            delta_val = 300.0

        # Initialize Tinker model parameters
        b_tinker: float
        d_tinker: float
        e_tinker: float
        f_tinker: float
        g_tinker: float

        if delta_val == 200:
            b_tinker, d_tinker, e_tinker, f_tinker, g_tinker = (
                0.482,
                1.97,
                1.0,
                0.51,
                1.228,
            )
        elif delta_val == 300:
            b_tinker, d_tinker, e_tinker, f_tinker, g_tinker = (
                0.466,
                2.06,
                0.99,
                0.48,
                1.310,
            )
        elif delta_val == 400:
            b_tinker, d_tinker, e_tinker, f_tinker, g_tinker = (
                0.494,
                2.30,
                0.93,
                0.48,
                1.403,
            )
        else:
            b_tinker, d_tinker, e_tinker, f_tinker, g_tinker = (
                0.466,
                2.06,
                0.99,
                0.48,
                1.310,
            )

        result = (
            b_tinker
            * (np.power(s / e_tinker, -d_tinker) + np.power(s, -f_tinker))
            * np.exp(-g_tinker / (s * s))
        )

    # Linear Diffusive Barrier
    elif model == 3 or model in ["2LDB"]:
        b_ldb: float
        d_ldb: float
        dv_ldb: float
        j_max_ldb: int
        if theta is not None:
            # Expecting theta to be a 1D NDArray of 4 floating-point parameters
            if isinstance(theta, np.ndarray):
                if not theta.shape == (4,):
                    raise ValueError(
                        "Theta for Linear Diffusive Barrier model must be a 1D"
                        "numpy array of 4 parameters [b, D, dv, J_max]."
                    )
                b_ldb, d_ldb, dv_ldb, j_max_ldb = (
                    float(theta[0]),
                    float(theta[1]),
                    float(theta[2]),
                    int(theta[3]),
                )
            else:
                raise ValueError(
                    "Theta for Linear Diffusive Barrier model must be a 1D "
                    "numpy array of 4 parameters [b, D, dv, J_max]."
                )
        else:
            b_ldb, d_ldb, dv_ldb, j_max_ldb = 0.0, 0.0, 2.71, 20

        result = np.zeros(len(s))
        dt: float = delta_c + dv_ldb

        for n in range(1, j_max_ldb + 1):
            result += (
                2.0
                * (1.0 + d_ldb)
                * np.exp(-b_ldb * b_ldb * s * s / (2.0 * (1.0 + d_ldb)))
                * np.exp(-b_ldb * delta_c / (1.0 + d_ldb))
                * (n * np.pi / (dt * dt))
                * s
                * s
                * np.sin(n * np.pi * delta_c / dt)
                * np.exp(
                    -n
                    * n
                    * np.pi
                    * np.pi
                    * s
                    * s
                    * (1.0 + d_ldb)
                    / (2.0 * dt * dt)
                )
            )

    return result


# Compute the halo mass function
def get_dn_dlnm(
    m: NDArray[np.floating],
    sigma: Optional[NDArray[np.floating]] = None,
    model: int | str = "PS",
    theta: Optional[float | NDArray[np.floating]] = None,
    delta_c: Optional[float] = None,
    omega_m0: float = 0.31,
    z: float = 0.0,
    k: Optional[NDArray[np.floating]] = None,
    pk: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.floating]:
    """
    Compute the logarithmic derivative of the halo mass function
    with respect to mass.

    :param m: Mass array.
    :type m: numpy.ndarray
    :param sigma: Variance of the density field. Fiducial value: None
    :type sigma: Optional[numpy.ndarray]
    :param model: HMF model to use (0: Press-Schechter, 1: Sheth-Tormen,
                  2: Tinker, 3: Linear Diffusive Barrier).
                  Can also be a string identifier ("PS", "ST", "Tinker",
                  "2LDB"). Fiducial value: "PS"
    :type model: Union[int, str]
    :param theta: Model parameters. For Sheth-Tormen: [a, b, p],
                  for Tinker: Delta,
                  for Linear Diffusive Barrier: [b, D, dv, J_max].
    :type theta: Optional[Union[float, np.ndarray]]
    :param delta_c: Critical density for collapse. Fiducial value: None
    :type delta_c: Optional[float]
    :param omega_m0: Omega matter at z=0. Fiducial value: 0.31
    :type omega_m0: float
    :param z: Redshift. Fiducial value: 0.0
    :type z: float
    :param k: Wavenumbers of the power spectrum (only if sigma is None).
              Fiducial value: None
    :type k: Optional[numpy.ndarray]
    :param pk: Power spectrum (required if sigma is None).
               Fiducial value: None
    :type pk: Optional[numpy.ndarray]

    :return: Halo mass function.
    :rtype: numpy.ndarray
    """
    rho_c = 2.775e11
    rho_m = omega_m0 * rho_c * np.power(1 + z, 3)

    # Compute sigma if it was not given
    if sigma is None:
        if k is None or pk is None:
            raise ValueError("sigma or (k, p) must be provided!")
        sigma = compute_sigma(k, pk, m=m, omega_m0=omega_m0, z=z)

    return (
        -fh(sigma, model, theta, delta_c, omega_m0, z)
        * rho_m
        / m
        * dlns_dlnm(m, sigma)
    )


# Halo bias of first order
def get_bh1(
    m: NDArray[np.floating],
    sigma: Optional[NDArray[np.floating]] = None,
    model: int | str = "PS",
    theta: Optional[float | NDArray[np.floating]] = None,
    delta_c: Optional[float] = None,
    omega_m0: float = 0.31,
    z: float = 0.0,
    k: Optional[NDArray[np.floating]] = None,
    pk: Optional[NDArray[np.floating]] = None,
    lagrangian: bool = False,
) -> NDArray[np.floating]:
    """
    Compute the first-order halo bias (b1).

    :param m: Mass array.
    :type m: numpy.ndarray
    :param sigma: Variance of the linear density field.
                  Fiducial value: None
    :type sigma: Optional[numpy.ndarray]
    :param model: HMF model to use (0: Press-Schechter, 1: Sheth-
                  Tormen, 2: Tinker, 3: Linear Diffusive Barrier).
                  Can also be a string identifier ("PS", "ST", "Tinker",
                  "2LDB"). Fiducial value: None
    :type model: Union[int, str]
    :param theta: Model parameters. For Sheth-Tormen: [a, b, p],
                  for Tinker: Delta,
                  for Linear Diffusive Barrier: [b, D, dv, J_max].
    :type theta: Optional[Union[float, np.ndarray]]
    :param delta_c: Critical density for collapse.
                    Fiducial value: None
    :type delta_c: Optional[float]
    :param omega_m0: Omega matter at z=0. Fiducial value: 0.31
    :type omega_m0: float
    :param z: Redshift. Fiducial value: 0.0
    :type z: float
    :param k: Wavenumbers of the power spectrum
              (required if sigma is None). Fiducial value: None
    :type k: Optional[numpy.ndarray]
    :param pk: Power spectrum (required if sigma is none).
               Fiducial value: None
    :type pk: Optional[numpy.ndarray]
    :param lagrangian: Whether to compute the Lagrangian bias.
    :type lagrangian: bool

    :return: First-order halo bias (b1).
    :rtype: numpy.ndarray
    """
    from scipy.special import binom

    # Compute sigma if it was not given
    if sigma is None:
        if k is None or pk is None:
            raise ValueError("sigma or (k, p) must be provided!")
        sigma = compute_sigma(k, pk, m=m, omega_m0=omega_m0, z=z)

    # Compute delta_c if it was not given
    if delta_c is None:
        delta_c = get_deltac(get_omz(omega_m0=omega_m0, z=np.array(z))).item()
    nu: NDArray[np.floating] = delta_c / sigma
    result: NDArray[np.floating] = np.zeros(len(sigma))

    # Press-Schechter
    if model == 0 or model in ["ps", "PS", "1SB"]:
        result = 1.0 + (nu * nu - 1.0) / delta_c

    # Sheth-Tormen
    elif model == 1 or model in ["ST", "st", "elliptical"]:
        a_st_param: float
        b_st_param: float
        p_st_param: float
        if theta is not None:
            # Expecting theta to be a 1D NDArray of 3 floating-point
            # parameters
            if isinstance(theta, np.ndarray):
                if not theta.shape == (3,):
                    raise ValueError(
                        "Theta for Sheth-Tormen model must be a 1D numpy"
                        " array of 3 parameters [a, b, p]."
                    )
                a_st_param, b_st_param, p_st_param = (
                    float(theta[0]),
                    float(theta[1]),
                    float(theta[2]),
                )
            else:
                raise ValueError(
                    "Theta for Sheth-Tormen model must be a 1D numpy"
                    " array of 3 parameters [a, b, p]."
                )
        else:
            a_st_param, b_st_param, p_st_param = 0.7, 0.4, 0.6

        coeff_a: float = 0.0
        for i in range(6):
            coeff_a += np.power(-1, i) * binom(p_st_param, i)

        result = (
            1.0
            + np.sqrt(a_st_param)
            * nu
            * nu
            / delta_c
            * (1.0 + b_st_param * np.power(a_st_param * nu * nu, -p_st_param))
            - 1.0
            / (
                np.sqrt(a_st_param)
                * delta_c
                * (1.0 + coeff_a * np.power(a_st_param * nu * nu, -p_st_param))
            )
        )

    # Tinker
    elif model == 2 or model in [
        "Tinker",
        "tinker",
        "TINKER",
    ]:
        delta_tinker_val: float
        if theta is not None:
            # Expecting theta to be a single scalar float or a 1-element
            # numpy array
            if isinstance(theta, np.ndarray):
                if theta.size != 1:
                    raise ValueError(
                        "Theta for Tinker model must be a single scalar"
                        " or an array of 1 element."
                    )
                delta_tinker_val = float(theta.item())
            else:  # theta is float
                delta_tinker_val = float(theta)
        else:
            delta_tinker_val = 300.0

        y: float = np.log10(delta_tinker_val)
        a_tinker_param: float = 1.0 + 0.24 * y * np.exp(-((4.0 / y) ** 4))
        b_tinker_param: float = 0.44 * y - 0.88
        c_tinker_param: float = 0.183
        d_tinker_param: float = 1.5
        e_tinker_param: float = (
            0.019 + 0.107 * y + 0.19 * np.exp(-((4.0 / y) ** 4))
        )
        f_tinker_param: float = 2.4

        result = (
            1.0
            - a_tinker_param
            * np.power(nu, b_tinker_param)
            / (
                np.power(nu, b_tinker_param)
                + np.power(delta_c, b_tinker_param)
            )
            + c_tinker_param * np.power(nu, d_tinker_param)
            + e_tinker_param * np.power(nu, f_tinker_param)
        )

    # Linear Diffusive Barrier
    elif model == 3 or model in ["2LDB"]:
        b_ldb: float
        d_ldb: float
        dv_ldb: float
        j_max_ldb: int
        if theta is not None:
            # Expecting theta to be a 1D NDArray of 4 floating-point
            # parameters
            if isinstance(theta, np.ndarray):
                if not theta.shape == (4,):
                    raise ValueError(
                        "Theta for Linear Diffusive Barrier model must be"
                        " a 1D numpy array of 4 parameters [b, D, dv,"
                        " J_max]."
                    )
                b_ldb, d_ldb, dv_ldb, j_max_ldb = (
                    float(theta[0]),
                    float(theta[1]),
                    float(theta[2]),
                    int(theta[3]),
                )
            else:
                raise ValueError(
                    "Theta for Linear Diffusive Barrier model must be"
                    " a 1D numpy array of 4 parameters [b, D, dv,"
                    " J_max]."
                )
        else:
            b_ldb, d_ldb, dv_ldb = 0.0, 0.0, 2.71
            j_max_ldb = 20
        result = np.zeros(len(sigma))
        temp = np.zeros(len(sigma))
        dt: float = delta_c + dv_ldb

        # Halos
        for n in range(1, int(j_max_ldb) + 1):
            result -= (
                (n * np.pi / (dt * dt))
                * np.sin(n * np.pi * delta_c / dt)
                * np.exp(
                    -n
                    * n
                    * np.pi
                    * np.pi
                    * sigma
                    * sigma
                    * (1.0 + d_ldb)
                    / (2.0 * dt * dt)
                )
                * (
                    np.power(np.tan(n * np.pi * delta_c / dt), -1.0)
                    * (n * np.pi / dt)
                    - b_ldb / (1.0 + d_ldb)
                )
            )

        for n in range(1, int(j_max_ldb) + 1):
            temp += (
                (n * np.pi / (dt * dt))
                * np.sin(n * np.pi * delta_c / dt)
                * np.exp(
                    -n
                    * n
                    * np.pi
                    * np.pi
                    * sigma
                    * sigma
                    * (1.0 + d_ldb)
                    / (2.0 * dt * dt)
                )
            )

        result = np.ones(len(sigma)) + result / temp

    # Convert to Lagrangian bias if needed
    if lagrangian:
        result -= 1.0

    return result


# Halo bias of second order
def get_bh2(
    m: NDArray[np.floating],
    sigma: Optional[NDArray[np.floating]] = None,
    model: int | str = "PS",
    theta: Optional[float | NDArray[np.floating]] = None,
    delta_c: Optional[float] = None,
    omega_m0: float = 0.31,
    z: float = 0.0,
    k: Optional[NDArray[np.floating]] = None,
    pk: Optional[NDArray[np.floating]] = None,
    lagrangian: bool = False,
    b_1: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.floating]:
    """
    Compute the second-order halo bias (b2).

    :param m: Mass array.
    :type m: numpy.ndarray
    :param sigma: Variance of the density field.
                  Fiducial value: None
    :type sigma: Optional[numpy.ndarray]
    :param model: HMF model to use (0: Press-Schechter, 1: Sheth-
                  Tormen, 2: Matteo, 3: Lazeyras). Can also be a string
                  identifier ("PS", "ST", "Matteo", "Lazeyras").
                  Fiducial value: "PS"
    :type model: Union[int, str]
    :param theta: Model parameters. For Sheth-Tormen: [a, b, p],
                  for Matteo: b1, for Lazeyras: b1.
    :type theta: Optional[Union[float, np.ndarray]]
    :param delta_c: Critical density for collapse.
                    Fiducial value: None
    :type delta_c: Optional[float]
    :param omega_m0: Omega matter at z=0. Fiducial value: 0.31
    :type omega_m0: float
    :param z: Redshift. Fiducial value: 0.0
    :type z: float
    :param k: Wavenumbers of the power spectrum
              (required if sigma is None). Fiducial value: None
    :type k: Optional[numpy.ndarray]
    :param pk: Power spectrum (required if sigma is None).
               Fiducial value: None
    :type pk: Optional[numpy.ndarray]
    :param lagrangian: Whether to compute the Lagrangian bias.
    :type lagrangian: bool
    :param b_1: First-order halo bias (used in Matteo's and
                Lazeyras's models). Fiducial value: None
    :type b_1: Optional[numpy.ndarray]

    :return: Array containing the second-order halo bias values
        (b2).
    :rtype: numpy.ndarray
    """
    from scipy.special import binom

    # Compute the variance of the linear density field if it was not given
    if sigma is None:
        if k is None or pk is None:
            raise ValueError("sigma or (k, pk) must be provided!")
        sigma = compute_sigma(k, pk, m=m, omega_m0=omega_m0, z=z)

    # Compute the critical density if it was not given
    if delta_c is None:
        delta_c = get_deltac(z=np.array(z), omega_m0=omega_m0).item()
    nu: NDArray[np.floating] = delta_c / sigma
    sigma_squared: NDArray[np.floating] = sigma**2
    resp: NDArray[np.floating] = np.zeros(len(sigma))

    # Press-Schechter
    if model == 0 or model in ["ps", "PS", "1SB"]:
        resp = np.power(nu * nu / delta_c, 2.0) - 3.0 * np.power(
            nu / delta_c, 2.0
        )

    # Sheth-Tormen
    elif model == 1 or model in ["ST", "st", "elliptical"]:
        a_st_param: float
        b_st_param: float
        p_st_param: float
        if theta is not None:
            if isinstance(theta, np.ndarray):
                if not theta.shape == (3,):
                    raise ValueError(
                        "Theta for Sheth-Tormen model must be a 1D numpy "
                        "array of 3 parameters [a, b, p]."
                    )
                a_st_param, b_st_param, p_st_param = (
                    float(theta[0]),
                    float(theta[1]),
                    float(theta[2]),
                )
            else:
                raise ValueError(
                    "Theta for Sheth-Tormen model must be a 1D numpy "
                    "array of 3 parameters [a, b, p]."
                )
        else:
            a_st_param, b_st_param, p_st_param = 0.7, 0.4, 0.6

        coeff_a: float = 0.0
        for i in range(6):
            coeff_a += np.power(-1, i) * binom(p_st_param, i)

        b_st_val: NDArray[np.floating] = (
            np.sqrt(a_st_param)
            * delta_c
            * (1.0 + b_st_param * np.power(a_st_param * nu * nu, -p_st_param))
        )
        bp_st_val: NDArray[np.floating] = (
            np.sqrt(a_st_param)
            * delta_c
            * (1.0 + coeff_a * np.power(a_st_param * nu * nu, -p_st_param))
        )

        resp = (
            np.power(b_st_val / sigma_squared, 2.0)
            - 1.0 / sigma_squared
            - 2.0 * b_st_val / (sigma_squared * bp_st_val)
        )

    # Matteo
    elif model == 2 or model in ["matteo", "Matteo"]:
        if b_1 is None:
            b_1 = get_bh1(
                m,
                sigma=sigma,
                model=2,  # Using Tinker model from get_bh1 for these recipes
                theta=330.0,  # Tinker's Delta parameter
                delta_c=delta_c,
                omega_m0=omega_m0,
                z=z,
                k=k,
                pk=pk,
                lagrangian=True,
            )
        resp = -0.09143 * b_1**3 + 0.7093 * b_1**2 - 0.2607 * b_1 - 0.3469

    # Lazeyras
    elif model == 3 or model in ["Lazeyras", "lazeyras"]:
        if b_1 is None:
            b_1 = get_bh1(
                m,
                sigma=sigma,
                model=2,  # Using Tinker model
                theta=330.0,  # Tinker's Delta parameter
                delta_c=delta_c,
                omega_m0=omega_m0,
                z=z,
                k=k,
                pk=pk,
                lagrangian=True,
            )
        resp = 0.412 - 2.143 * b_1 + 0.929 * b_1**2 + 0.008 * b_1**3

    if not lagrangian:
        if b_1 is None:
            b_1 = get_bh1(
                m,
                sigma=sigma,
                model=model,
                theta=theta,
                delta_c=delta_c,
                omega_m0=omega_m0,
                z=z,
                k=k,
                pk=pk,
                lagrangian=True,
            )
        resp = 4.0 / 21.0 * b_1 + 1.0 / 2.0 * resp

    return resp
    if not lagrangian:
        if b_1 is None:
            b_1 = get_bh1(
                m,
                sigma=sigma,
                model=model,
                theta=theta,
                delta_c=delta_c,
                omega_m0=omega_m0,
                z=z,
                k=k,
                pk=pk,
                lagrangian=True,
            )
        resp = 4.0 / 21.0 * b_1 + 1.0 / 2.0 * resp

    return resp


# Halo bias of third order
def get_bh3(
    mass: NDArray[np.floating],
    sigma: Optional[NDArray[np.floating]] = None,
    model: int | str = "PS",
    theta: Optional[float | NDArray[np.floating]] = None,
    delta_c: Optional[float] = None,
    omega_m0: float = 0.31,
    z: float = 0.0,
    k: Optional[NDArray[np.floating]] = None,
    pk: Optional[NDArray[np.floating]] = None,
    lagrangian: bool = False,
    b_s2: float = 0.0,
) -> NDArray[np.floating]:
    """
    Compute the third-order halo bias (b3).

    :param mass: Mass array.
    :type mass: numpy.ndarray
    :param sigma: Variance of the density field.
                  Fiducial value: None
    :type sigma: Optional[numpy.ndarray]
    :param model: HMF model to use (0: Press-Schechter,
                  1: Sheth-Tormen). Can also be a string identifier
                  ("PS", "ST"). Fiducial value: "PS"
    :type model: Union[int, str]
    :param theta: Model parameters. For Sheth-Tormen: [a, b, p].
    :type theta: Optional[Union[float, np.ndarray]]
    :param delta_c: Critical density for collapse.
                    Fiducial value: None
    :type delta_c: Optional[float]
    :param omega_m0: Omega matter at z=0. Fiducial value: 0.31
    :type omega_m0: float
    :param z: Redshift. Fiducial value: 0.0
    :type z: float
    :param k: Wavenumbers of the power spectrum
              (required if sigma is None). Fiducial value: None
    :type k: Optional[numpy.ndarray]
    :param pk: Power spectrum (required if sigma is None).
               Fiducial value: None
    :type pk: Optional[numpy.ndarray]
    :param lagrangian: Whether to compute the Lagrangian bias.
    :type lagrangian: bool
    :param b_s2: Second-order halo bias. Fiducial value: 0.0
    :type b_s2: float

    :return: Array containing the third-order halo bias values (b3).
    :rtype: numpy.ndarray
    """
    from scipy.special import binom

    # Compute the variance of the linear density grid if it was not given
    if sigma is None:
        if k is None or pk is None:
            raise ValueError("sigma or (k, pk) must be provided!")
        sigma = compute_sigma(k, pk, m=mass, omega_m0=omega_m0, z=z)

    # Compute the critical density if it was not given
    if delta_c is None:
        delta_c = get_deltac(z=np.array(z), omega_m0=omega_m0).item()
    nu: NDArray[np.floating] = delta_c / sigma
    sigma_squared: NDArray[np.floating] = sigma**2
    resp: NDArray[np.floating] = np.zeros(len(sigma))

    # Press-Schechter
    if model == 0 or model in ["ps", "PS", "1SB"]:
        b1_ps = (
            1.0 + (nu * nu - 1.0) / delta_c
        )  # This variable is assigned but not used later.
        resp = np.power(b1_ps, 3.0) - 3.0 * b1_ps / np.power(
            sigma_squared, 2.0
        )

    # Sheth-Tormen
    elif model == 1 or model in ["ST", "st", "elliptical"]:
        a_st_param: float
        b_st_param: float
        p_st_param: float
        if theta is not None:
            # Expecting theta to be a 1D NDArray of 3 floating-point parameters
            if isinstance(theta, np.ndarray):
                if not theta.shape == (3,):
                    raise ValueError(
                        "Theta for Sheth-Tormen model must be a 1D numpy "
                        "array of 3 parameters [a, b, p]."
                    )
                a_st_param, b_st_param, p_st_param = (
                    float(theta[0]),
                    float(theta[1]),
                    float(theta[2]),
                )
            else:
                raise ValueError(
                    "Theta for Sheth-Tormen model must be a 1D numpy "
                    "array of 3 parameters [a, b, p]."
                )
        else:
            a_st_param, b_st_param, p_st_param = 0.7, 0.4, 0.6

        coeff_a: float = 0.0
        for i in range(6):
            coeff_a += np.power(-1, i) * binom(p_st_param, i)

        b1_st = (  # This variable is assigned but not used later.
            1.0
            + np.sqrt(a_st_param)
            * nu
            * nu
            / delta_c
            * (1.0 + b_st_param * np.power(a_st_param * nu * nu, -p_st_param))
            - 1.0
            / (
                np.sqrt(a_st_param)
                * delta_c
                * (1.0 + coeff_a * np.power(a_st_param * nu * nu, -p_st_param))
            )
        )

        resp = (
            np.power(b1_st / sigma_squared, 3.0)  # b_st_val is undefined
            - 3.0
            * b1_st
            / np.power(sigma_squared, 2.0)  # b_st_val is undefined
            - 3.0
            * b1_st
            * b1_st
            / (
                sigma_squared * sigma_squared * b1_st
            )  # b_st_val, bp_st_val are undefined
            + 3.0 / (sigma_squared * b1_st)  # bp_st_val is undefined
        )

    if not lagrangian:
        b2: NDArray[np.floating] = get_bh2(
            mass,
            sigma=sigma,
            model=model,
            theta=theta,
            delta_c=delta_c,
            omega_m0=omega_m0,
            z=z,
            k=k,
            pk=pk,
            lagrangian=True,
        )
        resp = -1.0 / 2.0 * b2 + 1.0 / 6.0 * resp - 2.0 / 3.0 * b_s2

    return resp


# Get the mean number of central galaxies
def get_n_central_galaxies(
    m: NDArray[np.floating],
    log_m_min: float = 13.25424743,
    sig_log_m: float = 0.26461332,
) -> NDArray[np.floating]:
    """
    Compute the mean number of central galaxies as a function of halo mass.

    This function implements a simplified form of the satellite galaxy
    occupation distribution, focusing on the central galaxy. The current
    implementation uses a basic power-law relation and does not utilize
    `log_m_min` or `sig_log_m`.

    :param m: Halo mass array.
    :type m: numpy.ndarray
    :param log_m_min: The logarithm of the minimum halo mass for central
                      galaxy occupation. This parameter is not currently used
                      in the calculation. Fiducial value: 13.25424743
    :type log_m_min: float
    :param sig_log_m: The scatter in the logarithm of halo mass for central
                      galaxy occupation. This parameter is not currently used
                      in the calculation. Fiducial value: 0.26461332
    :type sig_log_m: float

    :return: The mean number of central galaxies for each halo mass.
    :rtype: numpy.ndarray
    """
    # Import the erf function
    from scipy.special import erf  # pyright: ignore[reportMissingImports]

    resp = 0.5 * (1.0 + erf(np.log(m) - log_m_min / sig_log_m))

    return resp


# Get the mean number of satellite galaxies
def get_n_satellite_galaxies(
    m: NDArray[np.floating],
    log_m0: float = 13.28383025,
    log_m1: float = 14.32465146,
    alpha: float = 1.00811277,
) -> NDArray[np.floating]:
    """
    Compute the mean number of satellite galaxies as a function of halo mass.

    This function implements a common functional form for the satellite
    galaxy occupation distribution. The number of satellites is modeled as
    a power-law that depends on the halo mass relative to two characteristic
    mass scales, `m0` and `m1`, and an exponent `alpha`.

    :param m: Halo mass array.
    :type m: numpy.ndarray
    :param log_m0: The logarithm of the characteristic halo mass `m0`.
                   Fiducial value: 13.28383025
    :type log_m0: float
    :param log_m1: The logarithm of the characteristic halo mass `m1`.
                   Fiducial value: 14.32465146
    :type log_m1: float
    :param alpha: The exponent determining the slope of the power-law
                  for satellite galaxy abundance. Fiducial value: 1.00811277
    :type alpha: float

    :return: The mean number of satellite galaxies for each halo mass.
    :rtype: numpy.ndarray
    """
    resp = np.power(
        (m - np.power(10.0, log_m0)) / np.power(10.0, log_m1), -alpha
    )

    return resp


# Get the mean number of galaxies
def get_n_galaxies(
    m: NDArray[np.floating],
    model: int | str = "PS",
    theta: Optional[float | NDArray[np.floating]] = None,
    delta_c: Optional[float] = None,
    omega_m0: float = 0.31,
    z: float = 0.0,
    k: Optional[NDArray[np.floating]] = None,
    pk: Optional[NDArray[np.floating]] = None,
    log_m_min: float = 13.25424743,
    sig_log_m: float = 0.26461332,
    log_m0: float = 13.28383025,
    log_m1: float = 14.32465146,
    alpha: float = 1.00811277,
) -> float:
    """
    Compute the mean number of galaxies by integrating the product of the Halo
    Occupation Distribution (HOD) and the Halo Mass Function (HMF) over mass.

    This function calculates the total galaxy number density by first
    determining the number of central and satellite galaxies per halo (HOD),
    then multiplying this by the logarithmic derivative of the halo mass
    function (dn/dlnM), and finally integrating the resulting product over
    the logarithmic mass range.

    :param m: Halo mass array.
    :type m: numpy.ndarray
    :param model: HMF model to use (0: Press-Schechter, 1: Sheth-Tormen,
                  2: Tinker, 3: Linear Diffusive Barrier).
                  Can also be a string ("PS", "ST", "Tinker", "2LDB").
                  Fiducial value: "PS"
    :type model: Union[int, str]
    :param theta: Model parameters for HMF. For Sheth-Tormen: [a, b, p],
                  for Tinker: Delta,
                  for Linear Diffusive Barrier: [b, D, dv, J_max].
    :type theta: Optional[Union[float, np.ndarray]]
    :param delta_c: Critical density for collapse for HMF. Fiducial value: None
    :type delta_c: Optional[float]
    :param omega_m0: Omega matter at z=0. Fiducial value: 0.31
    :type omega_m0: float
    :param z: Redshift. Fiducial value: 0.0
    :type z: float
    :param k: Wavenumbers of the power spectrum (only if sigma for HMF is None).
              Fiducial value: None
    :type k: Optional[numpy.ndarray]
    :param pk: Power spectrum (required if sigma for HMF is None).
               Fiducial value: None
    :type pk: Optional[numpy.ndarray]
    :param log_m_min: The logarithm of the minimum halo mass for central
                      galaxy occupation. Fiducial value: 13.25424743
    :type log_m_min: float
    :param sig_log_m: The scatter in the logarithm of halo mass for central
                      galaxy occupation. Fiducial value: 0.26461332
    :type sig_log_m: float
    :param log_m0: The logarithm of the characteristic halo mass `m0` for
                   satellite galaxy occupation. Fiducial value: 13.28383025
    :type log_m0: float
    :param log_m1: The logarithm of the characteristic halo mass `m1` for
                   satellite galaxy occupation. Fiducial value: 14.32465146
    :type log_m1: float
    :param alpha: The exponent for satellite galaxy abundance.
                  Fiducial value: 1.00811277
    :type alpha: float

    :return: The total mean number density of galaxies (N_gal).
    :rtype: float
    """
    # Import the Simpson's rule integration function
    from scipy.integrate import (
        simpson,  # pyright: ignore[reportUnknownVariableType]
    )

    # Compute the mean number of central galaxies for each halo mass
    n_central = get_n_central_galaxies(m, log_m_min, sig_log_m)

    # Compute the mean number of satellite galaxies for each halo mass
    n_satellite = get_n_satellite_galaxies(m, log_m0, log_m1, alpha)

    # Total number of galaxies per halo (sum of central and satellite HOD)
    n_gal_per_halo = n_central + n_satellite

    # Compute the halo mass function (dn/dlnm)
    # The get_dn_dlnm function will compute sigma internally if k and pk are provided.
    dn_dlnm_val = get_dn_dlnm(
        m=m,
        sigma=None,  # sigma is not explicitly passed to this function
        model=model,
        theta=theta,
        delta_c=delta_c,
        omega_m0=omega_m0,
        z=z,
        k=k,
        pk=pk,
    )

    # The integrand for the total number of galaxies is N_gal(m) * dn/dlnM.
    # We need to integrate this product with respect to ln(m).
    integrand = n_gal_per_halo * dn_dlnm_val

    # Compute the integral using Simpson's rule
    # The integration variable is ln(m)
    total_n_galaxies: float = simpson(integrand, x=np.log(m))

    return total_n_galaxies


def clpt_powers(
    k: NDArray[np.floating],
    pk: NDArray[np.floating],
    lambda_val: float = 0.7,
    k_max: float = 0.7,
    n_min: int = 5,
    n_max: int = 10,
    verbose: bool = False,
) -> dict[str, NDArray[np.floating]]:
    """
    Compute the power spectra of the operators using
    Convolution Lagrangian Perturbation Theory (CLPT).

    :param k: Wavenumber of the power spectrum.
    :type k: numpy.ndarray
    :param pk: Linear power spectrum.
    :type pk: numpy.ndarray
    :param lambda_val: Scale to be used to smooth the power spectrum.
                       Fiducial value: 0.7
    :type lambda_val: float
    :param k_max: Maximum wavenumber of the outputs. Fiducial value: 0.7
    :type k_max: float
    :param n_min: Minimum order used in the full computation of the terms.
                  Fiducial value: 5
    :type n_min: int
    :param n_max: Maximum order used in the Limber approximation of the terms.
                  Fiducial value: 10
    :type n_max: int
    :param verbose: Whether to output information in the C code.
                    Fiducial value: False
    :type verbose: bool

    :return: dictionary with the power spectra of the operators:
        - "k": Ndarray with the wavenumbers
        - "Plin": Ndarray with linear power spectrum used as input
        - "P11": Ndarray with result for the 11 power spectrum

    :rtype: dict
    """
    # Call the c function that compute the CLPT
    from .lib.analytical import (  # pyright: ignore[reportMissingImports]
        clpt_compute,  # pyright: ignore[reportUnknownVariableType]
    )

    # The type of `clpt_compute` is unknown without access to its definition.
    x = cast(
        dict[str, NDArray[np.floating]],
        clpt_compute(
            k.astype("float64"),
            pk.astype("float64"),
            np.float64(lambda_val),
            np.float64(k_max),
            np.int32(n_min),
            np.int32(n_max),
            np.int32(verbose),
        ),
    )

    return x


# Compute the generalized corraletion functions (Xi_lm)
def xi_lm(
    r: NDArray[np.floating],
    k: NDArray[np.floating],
    pk: NDArray[np.floating],
    lambda_val: float = 0.7,
    bessel_order: int = 0,
    m_k: int = 2,
    m_r: int = 0,
    num_points_gaussian_smooth: int = 11,
    alpha: float = 4.0,
    r_max: float = 1.0,
    verbose: bool = False,
) -> NDArray[np.floating]:
    """
    Compute the generalized correlation functions (Xi_lm).

    :param r: Radial distances for the output.
    :type r: numpy.ndarray
    :param k: Wavenumber of the power spectrum.
    :type k: numpy.ndarray
    :param pk: Linear power spectrum.
    :type pk: numpy.ndarray
    :param lambda_val: Scale to be used to smooth the power spectrum.
                       Fiducial value: 0.7
    :type lambda_val: float
    :param l: Order of the spherical Bessel's function. Fiducial value: 0
    :type l: int
    :param m_k: Power of k in the integral. Fiducial value: 2
    :type m_k: int
    :param m_r: Power of r in the integral. Fiducial value: 0
    :type m_r: int
    :param num_points_gaussian_smooth:
        Number of points used by the Gaussian smooth. Fiducial value: 11
    :type num_points_gaussian_smooth: int
    :param alpha: Value of alpha used by the Gaussian smooth.
                  Fiducial value: 4.0
    :type alpha: float
    :param r_max: Maximum radius for the smoothing. Fiducial value: 1.0
    :type r_max: float
    :param verbose: Whether to output information in the C code.
                    Fiducial value: False
    :type verbose: bool

    :return: The generalized correlation function
        :math: 'xi_{lm} = int dk k^{mk} r^{mr} P(k) j_l(kr)'.
    :rtype: numpy.ndarray
    """
    # Call the c function that compute the Xi_lm
    from .lib.analytical import (  # pyright: ignore[reportMissingImports]
        xilm_compute,  # pyright: ignore[reportUnknownVariableType]
    )

    x = cast(
        NDArray[np.floating],
        xilm_compute(
            r.astype("float64"),
            k.astype("float64"),
            pk.astype("float64"),
            np.float64(lambda_val),
            np.int32(bessel_order),
            np.int32(m_k),
            np.int32(m_r),
            np.int32(num_points_gaussian_smooth),
            np.float64(alpha),
            np.float64(r_max),
            np.int32(verbose),
        ),
    )

    return x


# Compute the 1-loop matter or galaxy power spectrum using classPT
def pgg_eftoflss(
    k: Optional[NDArray[np.floating]] = None,
    parameters: Optional[dict[str, float]] = None,
    b: Optional[NDArray[np.floating]] = None,
    cs: Optional[NDArray[np.floating]] = None,
    c: Optional[NDArray[np.floating]] = None,
    ir_resummation: bool = True,
    cb: bool = True,
    rsd: bool = True,
    ap: bool = False,
    om_fid: float = 0.31,
    z: float = 0.0,
    ls: list[int] | int = 0,
    pk_mult: Optional[dict[str, NDArray[np.floating]]] = None,
    fz: Optional[float] = None,
    out_mult: bool = False,
    h_units: bool = True,
    vectorized: bool = False,
) -> dict[str, NDArray[np.floating]]:
    """
    Compute the 1-loop matter or galaxy power spectrum using classPT.

    :param k: Wavenumbers of the power spectrum (need to run CLASS-PT).
              Fiducial value: None
    :type k: Optional[numpy.ndarray]
    :param parameters: Cosmological parameters used by CLASS.
                       Fiducial value: {}
    :type parameters: dict
    :param b: Values of the bias parameters (b1, b2, bG2, bGamma3, b4).
              Fiducial value: None
    :type b: Optional[numpy.ndarray]
    :param cs: Values of the stochastic parameters.2D for multitracers array.
               Fiducial value: None
    :type cs: Optional[numpy.ndarray]
    :param c: Values of the counterterms. 1D or 2D (multitracers) array.
              Fiducial value: None
    :type c: Optional[numpy.ndarray]
    :param ir_resummation: Option to do the IR resummation of the spectrum.
                           Fiducial value: True
    :type ir_resummation: bool
    :param cb: Option to add baryons. Fiducial value: True
    :type cb: bool
    :param RSD: Option to give the power spectrum in redshift space.
                Fiducial value: True
    :type RSD: bool
    :param ap: Option to use the Alcock-Paczynski (AP) effect.
               Fiducial value: False
    :type ap: bool
    :param om_fid: Omega matter fiducial for the AP correction.
                   Fiducial value: 0.31
    :type om_fid: float
    :param z: Redshift of the power spectrum. Fiducial value: 0.0
    :type z: float
    :param ls: The multipoles to be computed [0, 2, 4]. list or int.
    :type ls: Union[list[int], int]
    :param pk_mult: Multipoles of the power spectrum (don't need CLASS-PT).
                    Fiducial value: None
    :type pk_mult: Optional[numpy.ndarray]
    :param fz: Growth rate at redshift z. Fiducial value: None
    :type fz: Optional[float]
    :param out_mult: Whether output multipoles. Fiducial value: False
    :type out_mult: bool
    :param h_units: Whether to use h-units. Fiducial value: True
    :type h_units: bool
    :param vectorized: Whether to use vectorized operations.
                       Fiducial value: False
    :type vectorized: bool

    :return: dictionary with the computed power spectra.
    :rtype: dict
    """
    pk_multipoles: dict[str, NDArray[np.floating]]
    _f_z: float
    h: float  # Hubble constant in units of 100 km/s/Mpc

    if parameters is None:
        parameters = {}

    # Compute the power spectra using classPT
    if pk_mult is None:
        try:
            from classy import (  # pyright: ignore[reportMissingImports]
                Class,  # pyright: ignore[reportUnknownVariableType]
            )
        except ImportError:  # Use specific exception type
            raise ImportError(
                "classy module is not installed. "
                "Please install it using pip: pip install classy"
            ) from ImportError

        # Set the parameters
        model = cast(
            Any, Class()
        )  # Cast M to Any as classy.Class type is unknown without stubs
        params: dict[str, float | int] = {  # params can hold floats or ints
            "A_s": 2.089e-9,
            "n_s": 0.9649,
            "tau_reio": 0.052,
            "omega_b": 0.02237,
            "omega_cdm": 0.12,
            "h": 0.6736,
            "YHe": 0.2425,
            "N_ur": 2.0328,
            "N_ncdm": 1,
            "m_ncdm": 0.06,
        }
        for key, value in parameters.items():
            params[key] = value
        params["z_pk"] = z
        model.set(params)

        _cb_str: str = "Yes" if cb else "No"
        _ap_str: str = "Yes" if ap else "No"
        _ir_resummation_str: str = "Yes" if ir_resummation else "No"
        _rsd_str: str = "Yes" if rsd else "No"

        if rsd:
            model.set({
                "output": "mPk",
                "non linear": "PT",
                "IR resummation": _ir_resummation_str,
                "Bias tracers": "Yes",
                "cb": _cb_str,
                "RSD": _rsd_str,
                "AP": _ap_str,
                "Omfid": om_fid,
            })
        else:
            model.set({
                "output": "mPk",
                "non linear": "PT",
                "IR resummation": _ir_resummation_str,
                "Bias tracers": "Yes",
                "cb": _cb_str,
                "RSD": _rsd_str,
            })
        model.compute()

        # Compute the spectra of the basis
        if k is None:
            raise TypeError(
                "You have to give an array of k "
                "where to compute the power spectrum"
            )
        # k is guaranteed to be NDArray[np.floating] here
        _k: NDArray[np.floating] = k

        h = cast(float, model.h())
        _f_z = cast(float, model.scale_independent_growth_factor_f(z))
        m_mult: NDArray[np.floating] = cast(
            NDArray[np.floating], model.get_pk_mult(_k * h, z, len(_k))
        )

        # Save a dictionary with the spectra
        pk_multipoles = {}
        spectra_label = [
            "Id2d2",
            "Id2",
            "IG2",
            "Id2G2",
            "IG2G2",
            "FG2",
            "ctr",
            "lin",
            "1loop",
        ]
        spectra_ind = [1, 2, 3, 4, 5, 6, 10, 14, 0]
        for i in range(len(spectra_label)):
            pk_multipoles[spectra_label[i]] = m_mult[spectra_ind[i]]
        if rsd:
            spectra_label = [
                "FG2_0b1",
                "FG2_0",
                "FG2_2",
                "ctr_0",
                "ctr_2",
                "ctr_4",
            ]
            spectra_ind = [7, 8, 9, 11, 12, 13]
            for i in range(len(spectra_label)):
                pk_multipoles[spectra_label[i]] = m_mult[spectra_ind[i]]
            spectra_label = [
                "lin_0_vv",
                "lin_0_vd",
                "lin_0_dd",
                "lin_2_vv",
                "lin_2_vd",
                "lin_4_vv",
                "1loop_0_vv",
                "1loop_0_vd",
                "1loop_0_dd",
                "1loop_2_vv",
                "1loop_2_vd",
                "1loop_2_dd",
                "1loop_4_vv",
                "1loop_4_vd",
                "1loop_4_dd",
                "Idd2_0",
                "Id2_0",
                "IdG2_0",
                "IG2_0",
                "Idd2_2",
                "Id2_2",
                "IdG2_2",
                "IG2_2",
                "Id2_4",
                "IG2_4",
            ]
            for i in range(len(spectra_label)):
                pk_multipoles[spectra_label[i]] = m_mult[15 + i]

    else:
        pk_multipoles = pk_mult  # Assign the provided dictionary
        h = 1.0  # Default h when Class is not run
        if fz is None:
            _f_z = np.power(
                (om_fid * np.power(1.0 + z, 3.0))
                / (om_fid * np.power(1.0 + z, 3.0) + 1.0 - om_fid),
                0.5454,
            )
        else:
            _f_z = fz
        if rsd and len(pk_multipoles.keys()) < 10:
            raise ValueError(
                "There are not all spectra needed "
                "for the computations in redshift space"
            )

    # Get the number of tracers
    if b is None:
        raise TypeError(
            "You have to give an array with the values of the bias parameters"
        )
    _b: NDArray[np.floating] = b  # Cast b to its non-Optional type

    if cs is None:
        raise TypeError(
            "You have to give an array with the values "
            "of the stochastic parameters"
        )
    _cs: NDArray[np.floating] = cs  # Cast cs to its non-Optional type

    if c is None:
        raise TypeError(
            "You have to give an array with the values of the counterterms"
        )
    _c: NDArray[np.floating] = c  # Cast c to its non-Optional type

    n_tracers: int
    num_vectorizations: int

    # Get initial dimensions for clarity and safety checks
    b_ndim = _b.ndim
    cs_ndim = _cs.ndim
    c_ndim = _c.ndim

    # Step 1: Determine num_vectorizations and n_tracers from _b and reshape _b
    # The shape of _b (num_vectorizations, n_tracers, N_bias_params).
    # N_bias_params is implicitly 5 based on subsequent indexing
    # (e.g., _b[:, i, 0] to _b[:, i, 4]).
    if vectorized:
        num_vectorizations = _b.shape[0]
        if (
            b_ndim == 2
        ):  # _b was (num_vectorizations, N_bias_params) for 1 tracer
            n_tracers = 1
            # Reshape _b to (num_vectorizations, 1, N_bias_params)
            _b = _b.reshape((num_vectorizations, 1, _b.shape[1]))
        elif (
            b_ndim == 3
        ):  # _b was already (num_vectorizations, N_tracers, N_bias_params)
            n_tracers = _b.shape[1]
            # _b is already in the desired 3D shape, no reshape needed
        else:
            raise ValueError(
                "Invalid shape for bias parameters _b "
                f"when vectorized={vectorized}: {_b.shape}."
                "Expected 2D (num_vec, N_bias_params)"
                "or 3D (num_vec, N_tracers, N_bias_params)."
            )
    else:  # not vectorized
        num_vectorizations = 1
        if b_ndim == 1:  # _b was (N_bias_params,) for 1 tracer
            n_tracers = 1
            # Reshape _b to (1, 1, N_bias_params)
            _b = _b.reshape((1, 1, _b.shape[0]))
        elif b_ndim == 2:  # _b was (N_tracers, N_bias_params)
            n_tracers = _b.shape[0]
            # Reshape _b to (1, n_tracers, N_bias_params)
            _b = _b.reshape((1, n_tracers, _b.shape[1]))
        else:
            raise ValueError(
                "Invalid shape for bias parameters _b "
                f"when vectorized={vectorized}: {_b.shape}. "
                "Expected 1D (N_bias_params) or 2D (N_tracers, N_bias_params)."
            )

    # Step 2: Reshape _cs (num_vectorizations, n_tracers, N_stochastic_params)
    # The number of stochastic parameters (last dimension) depends on RSD.
    # For RSD, N_stochastic_params is 3 (from l_idx 0,1,2 in ctrs assignment).
    # In Real space, N_stochastic_params is implicitly 1 (from _cs[:, i]).
    if rsd:
        target_stoch_params_count = 3
        if cs_ndim == 1:  # (N_stoch_params,) for 1 tracer, not vectorized
            if n_tracers != 1:  # Should be consistent with _b
                raise ValueError(
                    f"Inconsistent n_tracers for _cs (shape={_cs.shape}) "
                    f"with _b's derived n_tracers ({n_tracers}) "
                    "when not vectorized and RSD."
                )
            _cs = _cs.reshape((1, 1, _cs.shape[0]))
        elif cs_ndim == 2:
            if vectorized:  # (num_vec, N_stoch_params) for 1 tracer
                if n_tracers != 1:  # Should be consistent with _b
                    raise ValueError(
                        f"Inconsistent n_tracers for _cs (shape={_cs.shape})"
                        f"with _b's derived n_tracers ({n_tracers})"
                        " when vectorized and RSD."
                    )
                _cs = _cs.reshape((num_vectorizations, 1, _cs.shape[1]))
            else:  # (N_tracers, N_stoch_params) for not vectorized
                if _cs.shape[0] != n_tracers:  # Should be consistent with _b
                    raise ValueError(
                        f"Inconsistent n_tracers for _cs (shape={_cs.shape})"
                        f"with _b's derived n_tracers ({n_tracers})"
                        " when not vectorized and RSD."
                    )
                _cs = _cs.reshape((1, n_tracers, _cs.shape[1]))
        elif (
            cs_ndim == 3
        ):  # (num_vec, n_tracers, N_stoch_params) - ideal shape
            if _cs.shape[0] != num_vectorizations or _cs.shape[1] != n_tracers:
                raise ValueError(
                    f"Inconsistent shape for _cs (shape={_cs.shape})"
                    f"with derived num_vectorizations ({num_vectorizations})"
                    f"or n_tracers ({n_tracers}) when RSD."
                )
            pass
        else:
            raise ValueError(
                f"Invalid shape for stochastic parameters _cs: {_cs.shape}. "
                "Expected 1D, 2D or 3D for RSD."
            )
        # Final check for the number of stochastic parameters
        if _cs.shape[2] != target_stoch_params_count:
            raise ValueError(
                f"Mismatched number of stochastic parameters for RSD:"
                f" expected {target_stoch_params_count}, got {_cs.shape[2]}"
                f". Full _cs shape: {_cs.shape}"
            )
    else:  # Real space
        # Target shape for _cs is (num_vectorizations, n_tracers).
        # Implicitly, there's 1 stochastic parameter per tracer.
        if cs_ndim == 0:  # Scalar
            if (
                n_tracers != 1 or num_vectorizations != 1
            ):  # Scalar implies 1 tracer, 1 vectorization
                raise ValueError(
                    f"Scalar _cs ({_cs.shape}) inconsistent with derived"
                    f" n_tracers ({n_tracers}) or num_vectorizations"
                    f" ({num_vectorizations}) when not RSD."
                )
            _cs = _cs.reshape((1, 1))
        elif cs_ndim == 1:  # (n_tracers,) or (num_vec,) or (1,)
            if (
                n_tracers == 1 and _cs.shape[0] == num_vectorizations
            ):  # (num_vec,) for 1 tracer, vectorized
                _cs = _cs.reshape((num_vectorizations, 1))
            elif (
                num_vectorizations == 1 and _cs.shape[0] == n_tracers
            ):  # (n_tracers,) for not vectorized
                _cs = _cs.reshape((1, n_tracers))
            else:
                raise ValueError(
                    f"Ambiguous shape for _cs when not RSD, 1D: {_cs.shape}. "
                    "Please ensure consistency with n_tracers "
                    "and vectorized flag."
                )
        elif (
            cs_ndim == 2
        ):  # (num_vec, n_tracers) or (n_tracers, 1) or (num_vec, 1)
            # If `_cs.shape[1]` is 1, it's (num_vec, 1) or (n_tracers, 1)
            # If `_cs.shape[1]` is n_tracers, it's (num_vec, n_tracers)
            # or (n_tracers, n_tracers if square)
            if vectorized:
                if _cs.shape[0] != num_vectorizations:
                    raise ValueError(
                        f"Inconsistent num_vectorizations for _cs "
                        f"(shape={_cs.shape}) with _b's derived "
                        f"num_vectorizations ({num_vectorizations}) "
                        "when vectorized and not RSD."
                    )
                if (
                    _cs.shape[1] == n_tracers
                    or _cs.shape[1] == 1
                    and n_tracers == 1
                ):  # (num_vec, n_tracers)
                    pass
                else:  # (num_vec, X) where X != n_tracers and X != 1
                    raise ValueError(
                        f"Invalid shape for _cs when vectorized, not "
                        f"RSD, 2D: {_cs.shape}. Expected "
                        f"(num_vec, n_tracers) or (num_vec, 1)."
                    )
            else:  # not vectorized: (n_tracers, 1)
                if _cs.shape[0] != n_tracers or _cs.shape[1] != 1:
                    raise ValueError(
                        f"Invalid shape for _cs when not vectorized, "
                        f"not RSD, 2D: {_cs.shape}. Expected "
                        f"(n_tracers, 1)."
                    )
                _cs = _cs.reshape((1, n_tracers))
        elif cs_ndim == 3:  # (num_vec, n_tracers, N_stoch_params_real)
            # N_stoch_params_real must be 1
            if (
                _cs.shape[0] != num_vectorizations
                or _cs.shape[1] != n_tracers
                or _cs.shape[2] != 1
            ):
                raise ValueError(
                    f"Invalid shape for _cs when not RSD, 3D: {_cs.shape}."
                    " Expected (num_vec, n_tracers, 1)."
                )
            pass
        else:
            raise ValueError(
                f"Invalid shape for stochastic parameters _cs: {_cs.shape}."
                " Expected 0D, 1D, 2D or 3D for Real space."
            )

    # Step 3: Reshape _c to (num_vectorizations, n_tracers, N_ct_params,
    # N_multipoles) or (num_vectorizations, n_tracers, N_ct_params)
    # The number of multipoles for counterterms (last dimension for RSD)
    # depends on RSD.
    # For RSD, N_multipoles is 3 (from pgg_l0, pgg_l2, pgg_l4 using
    # `_c_asserted[:, ind, i_ct, 1]` etc.).
    # For Real space, implicitly N_multipoles is 1.
    if rsd:
        target_c_multipoles = 3
        if c_ndim == 2:  # (N_ct_params, N_multipoles) for 1 tracer
            if n_tracers != 1 or num_vectorizations != 1:
                raise ValueError(
                    f"Inconsistent n_tracers/num_vec for _c (shape={_c.shape})"
                    " when not vectorized, 2D RSD."
                )
            _c = _c.reshape((1, 1, _c.shape[0], _c.shape[1]))
        elif c_ndim == 3:
            if vectorized:  # (num_vec, N_ct_params, N_multipoles) for 1 tracer
                if n_tracers != 1 or _c.shape[0] != num_vectorizations:
                    raise ValueError(
                        "Inconsistent n_tracers/num_vec for _c "
                        f"(shape={_c.shape}) when vectorized, 3D RSD."
                    )
                _c = _c.reshape((
                    num_vectorizations,
                    1,
                    _c.shape[1],
                    _c.shape[2],
                ))
            else:  # (n_tracers, N_ct_params, N_multipoles) for not vectorized
                if _c.shape[0] != n_tracers or num_vectorizations != 1:
                    raise ValueError(
                        "Inconsistent n_tracers/num_vec for _c "
                        f"(shape={_c.shape}) when not vectorized, 3D RSD."
                    )
                _c = _c.reshape((1, n_tracers, _c.shape[1], _c.shape[2]))
        elif c_ndim == 4:  # (num_vec, n_tracers, N_ct_params, N_multipoles)
            if _c.shape[0] != num_vectorizations or _c.shape[1] != n_tracers:
                raise ValueError(
                    f"Inconsistent shape for _c (shape={_c.shape}) with"
                    f" derived num_vectorizations ({num_vectorizations})"
                    f" or n_tracers ({n_tracers}) when RSD."
                )
            pass
        else:
            raise ValueError(
                f"Invalid shape for counterterms _c: {_c.shape}. "
                "Expected 2D, 3D or 4D for RSD."
            )
        # Final check for the number of counterterm multipoles
        if _c.shape[3] != target_c_multipoles:
            raise ValueError(
                f"Mismatched number of counterterm multipoles for RSD:"
                f" expected {target_c_multipoles}, got {_c.shape[3]}."
                f" Full _c shape: {_c.shape}"
            )
    else:  # Real space
        # Target shape for _c is (num_vectorizations, n_tracers, N_ct_params).
        if c_ndim == 1:  # (N_ct_params,) for 1 tracer, not vectorized
            if n_tracers != 1 or num_vectorizations != 1:
                raise ValueError(
                    f"Inconsistent n_tracers/num_vec for _c (shape={_c.shape})"
                    " when not vectorized, 1D Real."
                )
            _c = _c.reshape((1, 1, _c.shape[0]))
        elif c_ndim == 2:
            if vectorized:  # (num_vec, N_ct_params) for 1 tracer
                if n_tracers != 1 or _c.shape[0] != num_vectorizations:
                    raise ValueError(
                        f"Inconsistent n_tracers/num_vec for _c "
                        f"(shape={_c.shape}) when vectorized, 2D Real."
                    )
                _c = _c.reshape((num_vectorizations, 1, _c.shape[1]))
            else:  # (n_tracers, N_ct_params) for not vectorized
                if _c.shape[0] != n_tracers or num_vectorizations != 1:
                    raise ValueError(
                        f"Inconsistent n_tracers/num_vec for _c "
                        f"(shape={_c.shape}) when not vectorized, 2D Real."
                    )
                _c = _c.reshape((1, n_tracers, _c.shape[1]))
        elif c_ndim == 3:  # (num_vec, n_tracers, N_ct_params) - ideal shape
            if _c.shape[0] != num_vectorizations or _c.shape[1] != n_tracers:
                raise ValueError(
                    f"Inconsistent shape for _c (shape={_c.shape}) "
                    f"with derived num_vectorizations ({num_vectorizations}) "
                    f"or n_tracers ({n_tracers}) when Real."
                )
            pass
        else:
            raise ValueError(
                f"Invalid shape for counterterms _c: {_c.shape}. "
                "Expected 1D, 2D or 3D for Real space."
            )
    # Set all combinations of the bias parameters (b1, b2, bG2, bGamm3, b4)
    # The number of combinations is n_tracers * (n_tracers + 1) / 2
    num_combinations: int = n_tracers * (n_tracers + 1) // 2

    bias: NDArray[np.floating]
    ctrs: NDArray[np.floating]

    if rsd:
        bias = np.zeros([14, num_combinations, _b.shape[0]])
        ctrs = np.zeros([3, num_combinations, _cs.shape[0]])
        count = 0
        for i in range(n_tracers):
            for j in range(i):  # Off-diagonal elements (j < i)
                bias[:, count, :] = np.array(
                    [
                        (_b[:, i, 0] + _b[:, j, 0]) / 2.0,
                        _b[:, i, 0] * _b[:, j, 0],
                        _b[:, i, 1] * _b[:, j, 1],
                        (_b[:, i, 0] * _b[:, j, 1] + _b[:, i, 1] * _b[:, j, 0])
                        / 2.0,
                        (_b[:, i, 1] + _b[:, j, 1]) / 2.0,
                        (_b[:, i, 0] * _b[:, j, 2] + _b[:, i, 2] * _b[:, j, 0])
                        / 2.0,
                        (_b[:, i, 2] + _b[:, j, 2]) / 2.0,
                        (_b[:, i, 1] * _b[:, j, 2] + _b[:, i, 2] * _b[:, j, 1])
                        / 2.0,
                        _b[:, i, 2] * _b[:, j, 2],
                        (_b[:, i, 0] * _b[:, j, 3] + _b[:, i, 3] * _b[:, j, 0])
                        / 2.0,
                        (_b[:, i, 3] + _b[:, j, 3]) / 2.0,
                        (_b[:, i, 4] + _b[:, j, 4]) / 2.0,
                        (_b[:, i, 0] * _b[:, j, 4] + _b[:, i, 4] * _b[:, j, 0])
                        / 2.0,
                        (
                            _b[:, i, 0] ** 2 * _b[:, j, 4]
                            + _b[:, i, 4] * _b[:, j, 0] ** 2
                        )
                        / 2.0,
                    ],
                )
                for l_idx in range(
                    3
                ):  # Using l_idx to avoid conflict with `ls` parameter
                    ctrs[l_idx, count, :] = (
                        _cs[:, i, l_idx] * _b[:, j, 0]
                        + _cs[:, j, l_idx] * _b[:, i, 0]
                    ) / 2.0
                count += 1
            # Diagonal elements (j == i)
            bias[:, count, :] = np.array(
                [
                    _b[:, i, 0],
                    _b[:, i, 0] ** 2,
                    _b[:, i, 1] ** 2,
                    _b[:, i, 0] * _b[:, i, 1],
                    _b[:, i, 1],
                    _b[:, i, 0] * _b[:, i, 2],
                    _b[:, i, 2],
                    _b[:, i, 1] * _b[:, i, 2],
                    _b[:, i, 2] ** 2,
                    _b[:, i, 0] * _b[:, i, 3],
                    _b[:, i, 3],
                    _b[:, i, 4],
                    _b[:, i, 0] * _b[:, i, 4],
                    _b[:, i, 0] ** 2 * _b[:, i, 4],
                ],
            )
            for l_idx in range(3):
                ctrs[l_idx, count, :] = _cs[:, i, l_idx] * _b[:, i, 0]
            count += 1
    # (b1^2, b1*b2, b1*bG2, b1*bGamma3, b2^2, bG2^2, b2*bG2)
    else:  # Real space
        bias = np.zeros([7, num_combinations, _b.shape[0]])
        ctrs = np.zeros([num_combinations, _cs.shape[0]])
        count = 0
        for i in range(n_tracers):
            for j in range(i):  # Off-diagonal elements (j < i)
                bias[:, count, :] = np.array(
                    [
                        _b[:, i, 0] * _b[:, j, 0],
                        (_b[:, i, 0] * _b[:, j, 1] + _b[:, i, 1] * _b[:, j, 0])
                        / 2.0,
                        (_b[:, i, 0] * _b[:, j, 2] + _b[:, i, 2] * _b[:, j, 0])
                        / 2.0,
                        (_b[:, i, 0] * _b[:, j, 3] + _b[:, i, 3] * _b[:, j, 0])
                        / 2.0,
                        _b[:, i, 1] * _b[:, j, 1],
                        _b[:, i, 2] * _b[:, j, 2],
                        (_b[:, i, 1] * _b[:, j, 2] + _b[:, i, 2] * _b[:, j, 1])
                        / 2.0,
                    ],
                )
                ctrs[count, :] = (
                    _cs[:, i] * _b[:, j, 0] + _cs[:, j] * _b[:, i, 0]
                ) / 2.0
                count += 1
            # Diagonal elements (j == i)
            bias[:, count, :] = np.array(
                [
                    _b[:, i, 0] ** 2,
                    _b[:, i, 0] * _b[:, i, 1],
                    _b[:, i, 0] * _b[:, i, 2],
                    _b[:, i, 0] * _b[:, i, 3],
                    _b[:, i, 1] ** 2,
                    _b[:, i, 2] ** 2,
                    _b[:, i, 1] * _b[:, i, 2],
                ],
            )
            ctrs[count, :] = _cs[:, i] * _b[:, i, 0]
            count += 1

    # Define the functions to compute each power spectra
    def pgg(ind: int) -> NDArray[np.floating]:
        _k_asserted: NDArray[np.floating] = cast(
            NDArray[np.floating], k
        )  # k is guaranteed non-None here
        _c_asserted: NDArray[np.floating] = _c  # _c is already NDArray
        resp: NDArray[np.floating] = (
            bias[0, ind, :] * (pk_multipoles["lin"] + pk_multipoles["1loop"])
            + bias[1, ind, :] * pk_multipoles["Id2"]
            + 2.0 * bias[2, ind, :] * pk_multipoles["IG2"]
            + 2.0 * bias[2, ind, :] * pk_multipoles["FG2"]
            + 0.8 * bias[3, ind, :] * pk_multipoles["FG2"]
            + 0.25 * bias[4, ind, :] * pk_multipoles["Id2d2"]
            + bias[5, ind, :] * pk_multipoles["IG2G2"]
            + bias[6, ind, :] * pk_multipoles["Id2G2"]
        ) * np.power(h, 3.0 * h_units) + 2.0 * ctrs[ind, :] * pk_multipoles[
            "ctr"
        ] * np.power(h, h_units)
        for i_ct in range(
            len(_c_asserted[0, ind, :])
        ):  # Use i_ct to avoid shadowing outer 'i'
            resp += _c_asserted[:, ind, i_ct] * np.power(_k_asserted, 2 * i_ct)
        return resp

    def pgg_l0(ind: int) -> NDArray[np.floating]:
        _k_asserted: NDArray[np.floating] = cast(NDArray[np.floating], k)
        _c_asserted: NDArray[np.floating] = _c
        resp: NDArray[np.floating] = (
            (
                pk_multipoles["lin_0_vv"]
                + pk_multipoles["1loop_0_vv"]
                + bias[0, ind, :]
                * (pk_multipoles["lin_0_vd"] + pk_multipoles["1loop_0_vd"])
                + bias[1, ind, :]
                * (pk_multipoles["lin_0_dd"] + pk_multipoles["1loop_0_dd"])
                + 0.25 * bias[2, ind, :] * pk_multipoles["Id2d2"]
                + bias[3, ind, :] * pk_multipoles["Idd2_0"]
                + bias[4, ind, :] * pk_multipoles["Id2_0"]
                + bias[5, ind, :] * pk_multipoles["IdG2_0"]
                + bias[6, ind, :] * pk_multipoles["IG2_0"]
                + bias[7, ind, :] * pk_multipoles["Id2G2"]
                + bias[8, ind, :] * pk_multipoles["IG2G2"]
                + 2.0 * bias[5, ind, :] * pk_multipoles["FG2_0b1"]
                + 2.0 * bias[6, ind, :] * pk_multipoles["FG2_0"]
                + 0.8 * bias[9, ind, :] * pk_multipoles["FG2_0b1"]
                + 0.8 * bias[10, ind, :] * pk_multipoles["FG2_0"]
            )
            * np.power(h, 3.0 * h_units)
            + 2.0
            * ctrs[0, ind, :]
            * pk_multipoles["ctr_0"]
            * np.power(h, h_units)
            + _f_z**2
            * np.power(_k_asserted, 2.0)
            * 35
            / 8.0
            * pk_multipoles["ctr_4"]
            * (
                1.0 / 9.0 * bias[11, ind, :] * _f_z**2
                + 2.0 / 7.0 * _f_z * bias[12, ind, :]
                + 1.0 / 5.0 * bias[13, ind, :]
            )
            * np.power(h, h_units)
        )
        for i_ct in range(len(_c_asserted[0, ind, :, 1])):
            resp += _c_asserted[:, ind, i_ct, 1] * np.power(
                _k_asserted, 2 * i_ct
            )

        return resp

    def pgg_l2(ind: int) -> NDArray[np.floating]:
        _k_asserted: NDArray[np.floating] = cast(NDArray[np.floating], k)
        _c_asserted: NDArray[np.floating] = _c
        resp: NDArray[np.floating] = (
            (
                pk_multipoles["lin_2_vv"]
                + pk_multipoles["1loop_2_vv"]
                + bias[0, ind, :]
                * (pk_multipoles["lin_2_vd"] + pk_multipoles["1loop_2_vd"])
                + bias[1, ind, :] * pk_multipoles["1loop_2_dd"]
                + bias[3, ind, :] * pk_multipoles["Idd2_2"]
                + bias[4, ind, :] * pk_multipoles["Id2_2"]
                + bias[5, ind, :] * pk_multipoles["IdG2_2"]
                + bias[6, ind, :] * pk_multipoles["IG2_2"]
                + (2.0 * bias[6, ind, :] + 0.8 * bias[10, ind, :])
                * pk_multipoles["FG2_2"]
            )
            * np.power(h, 3.0 * h_units)
            + 2.0
            * ctrs[1, ind, :]
            * pk_multipoles["ctr_2"]
            * np.power(h, h_units)
            + _f_z**2
            * np.power(_k_asserted, 2.0)
            * 35
            / 8.0
            * pk_multipoles["ctr_4"]
            * (
                70.0 * bias[11, ind, :] * _f_z**2
                + 165.0 * _f_z * bias[12, ind, :]
                + 99.0 * bias[13, ind, :]
            )
            * (4.0 / 693.0)
            * np.power(h, h_units)
        )
        for i_ct in range(len(_c_asserted[0, ind, :, 1])):
            resp += _c_asserted[:, ind, i_ct, 1] * np.power(
                _k_asserted, 2 * i_ct
            )

        return resp

    def pgg_l4(ind: int) -> NDArray[np.floating]:
        _k_asserted: NDArray[np.floating] = cast(NDArray[np.floating], k)
        _c_asserted: NDArray[np.floating] = _c
        resp: NDArray[np.floating] = (
            (
                pk_multipoles["lin_4_vv"]
                + pk_multipoles["1loop_4_vv"]
                + bias[0, ind, :] * pk_multipoles["1loop_4_vd"]
                + bias[1, ind, :] * pk_multipoles["1loop_4_dd"]
                + bias[4, ind, :] * pk_multipoles["Id2_4"]
                + bias[6, ind, :] * pk_multipoles["IG2_4"]
            )
            * np.power(h, 3.0 * h_units)
            + 2.0
            * ctrs[2, ind, :]
            * pk_multipoles["ctr_4"]
            * np.power(h, h_units)
            + _f_z**2
            * np.power(_k_asserted, 2.0)
            * 35
            / 8.0
            * pk_multipoles["ctr_4"]
            * (
                210.0 * bias[11, ind, :] * _f_z**2
                + 390.0 * _f_z * bias[12, ind, :]
                + 143.0 * bias[13, ind, :]
            )
            * (8.0 / 5005.0)
            * np.power(h, h_units)
        )
        for i_ct in range(len(_c_asserted[0, ind, :, 2])):
            resp += _c_asserted[:, ind, i_ct, 2] * np.power(
                _k_asserted, 2 * i_ct
            )

        return resp

    # Compute the spectra and save in the dictionary
    x: dict[str, NDArray[np.floating]] = {}

    # Handle ls being an int or list for iteration
    _ls_list: list[int] = [ls] if isinstance(ls, int) else ls

    # k is guaranteed non-None here from previous checks or the pk_mult branch
    _k_len: int = len(cast(NDArray[np.floating], k))

    if rsd:
        if 0 in _ls_list:
            p_l0: NDArray[np.floating] = np.zeros(
                [
                    _b.shape[0],
                    num_combinations,
                    _k_len,
                ],
            )
            ind = 0
            for i in range(n_tracers):
                for _ in range(i + 1):
                    p_l0[:, ind, :] = pgg_l0(ind)
                    ind += 1
            x["Pgg_l0"] = p_l0
        if 2 in _ls_list:
            p_l2: NDArray[np.floating] = np.zeros(
                [
                    _b.shape[0],
                    num_combinations,
                    _k_len,
                ],
            )
            ind = 0
            for i in range(n_tracers):
                for _ in range(i + 1):
                    p_l2[:, ind, :] = pgg_l2(ind)
                    ind += 1
            x["Pgg_l2"] = p_l2
        if 4 in _ls_list:
            p_l4: NDArray[np.floating] = np.zeros(
                [
                    _b.shape[0],
                    num_combinations,
                    _k_len,
                ],
            )
            ind = 0
            for i in range(n_tracers):
                for _ in range(i + 1):
                    p_l4[:, ind, :] = pgg_l4(ind)
                    ind += 1
            x["Pgg_l4"] = p_l4
    else:
        p_real: NDArray[np.floating] = np.zeros([
            _b.shape[0],
            num_combinations,
            _k_len,
        ])
        ind = 0
        for i in range(n_tracers):
            for _ in range(i + 1):
                p_real[:, ind, :] = pgg(ind)
                ind += 1
        x["Pgg"] = p_real

    # Output the spectra
    if out_mult:
        # pk_multipoles is guaranteed to be dict[str, NDArray[np.floating]]
        for key in pk_multipoles:
            if (
                key == "ctr"
                or key == "ctr_0"
                or key == "ctr_2"
                or key == "ctr_4"
            ):
                x[key] = pk_multipoles[key] * np.power(h, h_units)
            else:
                x[key] = pk_multipoles[key] * np.power(h, 3.0 * h_units)

    return x
