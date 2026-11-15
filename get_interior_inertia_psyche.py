r"""
Moments of inertia of 16 Psyche from degree-2 gravity and rotational dynamics, by MCMC,
compared with Ermakov (2026), PSJ 7:170, "Interrogating 16 Psyche's Interior with
Rotational Dynamics and Gravity Field".

Standalone: no orbit propagation, no MCMCModel. Only the figures reuse the house style of
MCMCPlots.py (configure, plot_corner, save_figure, write_output).

Shape
-----
Shepard et al. (2021) shape of Psyche as a degree-10 spherical-harmonic expansion of the
surface radius r(theta, phi) (Zenodo 10.5281/zenodo.12522382, file Psyche-Shepard2021.sh;
4pi-normalized real harmonics, metres, lines "n, m, C_nm, S_nm"). The body is star-shaped
about the expansion origin, so every mass integral of a region {r < R(u)} is an angular
integral with an analytic radial part (unit density):

    mass = int R^3 / 3 dOmega,  first = int u R^4 / 4 dOmega,  second = int u u^T R^5 / 5 dOmega,

evaluated by Gauss-Legendre quadrature in cos(theta) times a uniform rule in phi.
Normalization as in the paper: A_bar = A / (M R0^2), R0 = 109 km, and

    J2 = C_bar - (A_bar + B_bar) / 2,   C22 = (B_bar - A_bar) / 4,   H = J2 / C_bar,
    I_bar = (A_bar + B_bar + C_bar) / 3 = J2 (1/H - 2/3).

Steps (each one compared with the paper in summary.txt)
------------------------------------------------------
1. Homogeneous body: A_bar, B_bar, C_bar, J2, C22, H, I_bar against Table 1 (Shepard row).
2. Two-layer endmembers (paper Fig. 2): core of radius r2 and density rho2; mantle density
   rho1 from the mean density RHO_MEAN. (A) spherical core truncated by the surface where
   exposed (r2 > r_min), (B) core = outer shape rescaled to the volume-equivalent radius r2.
   Minimum H and I_bar with rho1 >= RHO1_MIN (paper: 0.19 and 0.30); H unchanged for (B).
3. MCMC (a), model-agnostic (paper Sec. 5): theta = (A_bar, B_bar, C_bar), uniform prior on
   75-110 % of the homogeneous values (A <= B <= C, A + B >= C); data J2, C22
   (sigma/J2 = 0.004 %, Zuber et al. 2022) and H from precession / nutation, for a sweep of
   sigma_H / H. The posterior sigma of I_bar is compared with Eq. (13),
   sigma_I = C_bar sigma_H / H, and with the paper's 1.8 % for sigma_I = 0.01.
4. MCMC (b), physical two-layer model: theta = (r2, rho2, w), with the core boundary
   R_c(u) = min(R(u), r2 [(1 - w) + w R(u) / R_v]) blending endmember (A) (w = 0) into
   endmember (B) (w = 1); data J2, C22 (with a model-error floor) and H. Sampled in
   phi = (ln r2, ln(rho2 - rho_mean), w), which straightens the J2/C22 ridge
   (rho2 - rho_mean) r2^5 ~ const; the Jacobian keeps the prior uniform in theta.
5. Precession (paper Eq. 1): obliquity, precession period and pole drift rate of the
   homogeneous body from the spin pole and the heliocentric orbit, against Table 3 and the
   1.1 My period; the drift rate scales with H (paper Eq. 12), which sets sigma_H / H.

The synthetic data are the TRUTH two-layer model plus Gaussian noise (seed SEED_DATA).

Outputs (results/<script name>_<YYMMDDHHMMSS>/)
-----------------------------------------------
chain.npz (every chain, data and settings; replot with LOAD_CHAIN), summary.txt (paper vs
this script), and one PDF per figure: endmember maps, sigma_I vs sigma_H / H, mean moment
of inertia marginals, moment corner (a), traces and corners (b), core cross-sections.
"""

from __future__ import annotations

import io
import time
from datetime import datetime
from math import factorial
from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import least_squares
from scipy.special import lpmv

import MCMCPlots

MCMCPlots.SHOW_FIGS = True  # plt.show() after each figure is saved

HERE = Path(__file__).resolve().parent

# ============================================================
# Settings
# ============================================================

# --- Shape and bulk properties ---
SHAPE_SH = HERE / "data" / "shape_models" / "Psyche-Shepard2021.sh"  # SH radius model [m]
BODY = "Psyche"
R0 = 109.0e3  # normalization radius of the paper [m]
RHO_MEAN = 4172.0  # mean density [kg/m^3] (Shepard et al. 2021; paper Sec. 1)
RHO1_MIN = 2000.0  # minimum mantle (crust) density [kg/m^3] (paper Sec. 4.1)
N_MU, N_PHI = 64, 128  # angular quadrature: Gauss-Legendre nodes in cos(theta) x uniform phi

# --- Endmember maps (paper Fig. 2) ---
GRID_R2 = (5.0, 110.0, 106)  # core radius r2 [km]: start, stop, count
GRID_RHO2 = (RHO_MEAN, 12000.0, 80)  # core density rho2 [kg/m^3]: start, stop, count
RHO2_METAL = 8000.0  # Fe-Ni metal: second bound reported for the minimum I_bar

# --- Paper values (Ermakov 2026) ---
PAPER_HOM = dict(A=0.3566, B=0.4430, C=0.5524, J2=0.1526, C22=0.0216, H=0.2762)  # Table 1
PAPER_RMIN = 79.0  # [km]
PAPER_MIN_H, PAPER_MIN_I = 0.19, 0.30  # spherical core, rho1 >= 2000
PAPER_H_RANGE = (0.05, 0.51)  # grid of A, B, C at 75-110 % of homogeneous
PAPER_SIGMA_H_REL = (1.8, 2.6)  # [%] sigma_H / H for sigma_I = 0.01 (Sec. 5)
PAPER_POLE_RATE = (886.89, 280.99)  # d alpha / dt, d delta / dt [mas/yr] (Table 3)
PAPER_PREC_PERIOD = 1.1  # [My]
PAPER_OBLIQUITY = (93.2, 99.4)  # obliquity range over 10 My [deg]

# --- Rotation and orbit (precession, paper Eq. 1) ---
POLE_RA, POLE_DEC = 36.0, 6.0  # spin pole, ICRF [deg] (Shepard et al. 2021; Table 3)
SPIN_RATE = 2059.129428  # W rate [deg/day] (Table 3)
ORBIT = dict(a=2.9234, e=0.1340, i=3.0969, node=150.032)  # [AU, -, deg, deg] ecliptic J2000
#   approximate osculating elements of 16 Psyche (JPL Small-Body Database)
GM_SUN = 1.32712440018e20  # [m^3/s^2]
AU = 1.495978707e11  # [m]
OBLIQ_J2000 = 23.4392911  # obliquity of the ecliptic [deg]

# --- Truth (synthetic data): two-layer model, theta = (r2 [km], rho2 [kg/m^3], w) ---
TRUTH = np.array([70.0, 7000.0, 0.5])

# --- Measurement noise ---
SEED_DATA = 2026  # noise realization of the synthetic data
SIGMA_J2_REL = 4.0e-5  # sigma / value for J2 and C22 (0.004 %, Zuber et al. 2022)
SIGMA_H_REL = 0.018  # sigma_H / H of MCMC (b) (paper: enough for sigma_I = 0.01)
SIGMA_H_SWEEP = [0.005, 0.01, 0.018, 0.026, 0.05, 0.10]  # sigma_H / H of the MCMC (a) runs
SIGMA_MODEL_REL = 1.0e-3  # MCMC (b): model-error floor on J2, C22 (shape, quadrature)

# --- Priors ---
PRIOR_FRAC = (0.75, 1.10)  # MCMC (a): each moment within this fraction of homogeneous
PRIOR_B_LO = np.array([10.0, RHO_MEAN, 0.0])  # MCMC (b): r2 [km], rho2 [kg/m^3], w
PRIOR_B_HI = np.array([110.0, RHO2_METAL, 1.0])

# --- MCMC ---
N_WALKERS = 32
N_STEPS_A = 6000  # per sigma_H / H of the sweep (and the run without H)
N_STEPS_B = 30_000
BURN_IN = None  # None: 5 x max autocorrelation time
THIN = None  # None: max(1, min autocorrelation time / 2)
LOAD_CHAIN = None  # e.g. "results/get_interior_inertia_psyche_<stamp>/chain.npz" (replot only)

N_DERIVED = 4000  # posterior samples of (b) mapped to derived quantities
N_PRIOR = 20_000  # prior samples for the prior overlays

SYM_A = [r"\bar{A}", r"\bar{B}", r"\bar{C}", r"\bar{I}"]
SYM_B = [r"r_2", r"\rho_2", r"w"]
UNIT_B = ["km", "kg/m$^3$", ""]
DER_NAMES = ["I_bar", "C_bar", "H", "J2", "rho1", "r_c"]
DER_SYM = [r"\bar{I}", r"\bar{C}", r"H", r"J_2", r"\rho_1", r"r_{\mathrm{c}}"]
DER_UNIT = ["", "", "", "", "kg/m$^3$", "km"]


# ============================================================
# Shape model (spherical-harmonic radius) and mass integrals
# ============================================================


def load_sh_shape(path):
    """
    Read a spherical-harmonic radius model.

    Parameters
    ----------
    path : str or Path
        Text file with one "n, m, C_nm, S_nm" line per coefficient (4pi-normalized, m).

    Returns
    -------
    C, S : ndarray, shape (L + 1, L + 1)
        Cosine and sine coefficients [m], indexed [n, m].
    """
    d = np.loadtxt(path, delimiter=",", ndmin=2)
    L = int(d[:, 0].max())
    C, S = np.zeros((L + 1, L + 1)), np.zeros((L + 1, L + 1))
    for n, m, c, s in d:
        C[int(n), int(m)], S[int(n), int(m)] = c, s
    return C, S


def sh_radius(C, S, u):
    """
    Surface radius along unit directions from 4pi-normalized real spherical harmonics.

    Parameters
    ----------
    C, S : ndarray, shape (L + 1, L + 1)
        Coefficients of load_sh_shape [m].
    u : ndarray, shape (N, 3)
        Unit directions in the shape frame.

    Returns
    -------
    ndarray, shape (N,)
        R(u) [m]. The Condon-Shortley phase of scipy's lpmv is removed.
    """
    mu = np.clip(u[:, 2], -1.0, 1.0)
    phi = np.arctan2(u[:, 1], u[:, 0])
    r = np.zeros(len(u))
    for n in range(C.shape[0]):
        for m in range(n + 1):
            norm = np.sqrt((2 - (m == 0)) * (2 * n + 1) * factorial(n - m) / factorial(n + m))
            p = norm * (-1) ** m * lpmv(m, n, mu)
            r += p * (C[n, m] * np.cos(m * phi) + S[n, m] * np.sin(m * phi))
    return r


class AngularGrid:
    """
    Angular quadrature of a star-shaped body and its unit-density mass integrals.

    Parameters
    ----------
    C, S : ndarray
        Shape coefficients (load_sh_shape).
    n_mu, n_phi : int
        Gauss-Legendre nodes in cos(theta) and uniform nodes in phi.

    Attributes
    ----------
    u : ndarray, shape (N, 3)
        Quadrature directions.
    w : ndarray, shape (N,)
        Solid-angle weights (sum 4 pi).
    R : ndarray, shape (N,)
        Surface radius along u [m].
    outer : tuple
        (volume [m^3], first moment [m^4], second moment [m^5]) of the whole shape.
    r_v : float
        Volume-equivalent radius [m].
    """

    def __init__(self, C, S, n_mu, n_phi):
        mu, wmu = np.polynomial.legendre.leggauss(n_mu)
        phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
        MU, PHI = (a.ravel() for a in np.meshgrid(mu, phi, indexing="ij"))
        st = np.sqrt(1.0 - MU**2)
        self.u = np.column_stack([st * np.cos(PHI), st * np.sin(PHI), MU])
        self.w = np.outer(wmu, np.full(n_phi, 2.0 * np.pi / n_phi)).ravel()
        self._uu = np.einsum("ni,nj->nij", self.u, self.u).reshape(-1, 9)
        self.R = sh_radius(C, S, self.u)
        self.outer = self.region(self.R)
        self.r_v = (3.0 * self.outer[0] / (4.0 * np.pi)) ** (1.0 / 3.0)

    def region(self, Rr):
        """
        Unit-density mass integrals of the star-shaped region {r < Rr(u)}.

        Parameters
        ----------
        Rr : ndarray, shape (N,)
            Boundary radius along the quadrature directions [m].

        Returns
        -------
        vol : float
            Volume [m^3].
        first : ndarray, shape (3,)
            int x dV [m^4].
        second : ndarray, shape (3, 3)
            int x x^T dV [m^5].
        """
        r3 = self.w * Rr**3
        return r3.sum() / 3.0, (r3 * Rr) @ self.u / 4.0, ((r3 * Rr**2) @ self._uu).reshape(3, 3) / 5.0


def inertia_quantities(mass, first, second):
    """
    Normalized principal moments and degree-2 quantities about the centre of mass.

    Parameters
    ----------
    mass : float
        [kg].
    first : ndarray, shape (3,)
        int x dm [kg m].
    second : ndarray, shape (3, 3)
        int x x^T dm [kg m^2].

    Returns
    -------
    dict
        A, B, C (normalized by M R0^2, ascending), J2, C22, H, I (mean moment), com [m].
    """
    com = first / mass
    Sc = second - mass * np.outer(com, com)
    a, b, c = np.linalg.eigvalsh(np.trace(Sc) * np.eye(3) - Sc) / (mass * R0**2)
    J2 = c - 0.5 * (a + b)
    return dict(A=a, B=b, C=c, J2=J2, C22=0.25 * (b - a), H=J2 / c, I=(a + b + c) / 3.0, com=com)


class TwoLayer:
    """
    Two-layer Psyche: core of density rho2 inside a mantle of density rho1.

    The core boundary along u is R_c(u) = min(R(u), r2 [(1 - w) + w R(u) / R_v]):
    w = 0 is a sphere of radius r2 truncated by the surface (endmember A), w = 1 the outer
    shape rescaled to the volume-equivalent radius r2 (endmember B). rho1 follows from the
    mean density.

    Parameters
    ----------
    grid : AngularGrid
    rho_mean : float
        Mean density [kg/m^3].
    """

    def __init__(self, grid, rho_mean):
        self.g, self.rho_mean = grid, rho_mean

    def core_radius(self, r2_m, w, R=None):
        """Core boundary [m] along the directions of R (default: the quadrature grid)."""
        R = self.g.R if R is None else R
        return np.minimum(R, r2_m * ((1.0 - w) + w * R / self.g.r_v))

    def __call__(self, r2_km, rho2, w):
        """
        Inertia quantities of one interior.

        Parameters
        ----------
        r2_km : float
            Core radius [km].
        rho2 : float
            Core density [kg/m^3].
        w : float
            Shape blend, 0 (sphere) to 1 (rescaled outer shape).

        Returns
        -------
        dict
            inertia_quantities plus rho1 [kg/m^3] and r_c (volume-equivalent core radius,
            km); rho1 is -inf when the core fills the body.
        """
        V, F, S2 = self.g.outer
        v, f, s = self.g.region(self.core_radius(1e3 * r2_km, w))
        if V - v <= 1e-9 * V:
            return dict(rho1=-np.inf)
        rho1 = (self.rho_mean * V - rho2 * v) / (V - v)
        d = inertia_quantities(
            rho1 * V + (rho2 - rho1) * v, rho1 * F + (rho2 - rho1) * f, rho1 * S2 + (rho2 - rho1) * s
        )
        d["rho1"] = rho1
        d["r_c"] = (3.0 * v / (4.0 * np.pi)) ** (1.0 / 3.0) / 1e3
        return d


# ============================================================
# Endmember maps and the moment-of-inertia grid
# ============================================================


def endmember_map(model, w, r2s, rho2s):
    """
    H, J2, I_bar and rho1 on a (r2, rho2) grid for a fixed core shape.

    Returns
    -------
    dict of ndarray, shape (len(r2s), len(rho2s))
        Keys H, J2, I, rho1.
    """
    out = {k: np.full((len(r2s), len(rho2s)), np.nan) for k in ("H", "J2", "I", "rho1")}
    for i, r2 in enumerate(r2s):
        for j, rho2 in enumerate(rho2s):
            d = model(r2, rho2, w)
            if np.isfinite(d["rho1"]):
                for k in out:
                    out[k][i, j] = d[k]
    return out


def masked_extreme(field, ok, fn):
    """Extreme value of `field` where `ok` and the (i, j) index where it is reached."""
    vals = np.where(ok, field, np.nan)
    k = np.unravel_index(fn(vals), vals.shape)
    return vals[k], k


def moment_grid_H_range(hom, frac=PRIOR_FRAC, n=8):
    """
    Range of H on the paper's grid of A, B, C at frac x homogeneous (n = 8: 5 % steps) with C
    the largest.

    Returns
    -------
    (float, float)
        Minimum and maximum H.
    """
    f = np.linspace(*frac, n)
    a, b, c = np.meshgrid(hom["A"] * f, hom["B"] * f, hom["C"] * f, indexing="ij")
    ok = (c > a) & (c > b)
    H = (c - 0.5 * (a + b)) / c
    return H[ok].min(), H[ok].max()


# ============================================================
# Precession (paper Eq. 1)
# ============================================================


def unit_radec(ra_deg, dec_deg):
    """Unit vector of a right ascension and declination [deg]."""
    a, d = np.radians(ra_deg), np.radians(dec_deg)
    return np.array([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)])


def orbit_normal_icrf(incl_deg, node_deg):
    """Orbit normal (ecliptic J2000 inclination and node) in the ICRF equatorial frame."""
    i, O = np.radians(incl_deg), np.radians(node_deg)
    k = np.array([np.sin(i) * np.sin(O), -np.sin(i) * np.cos(O), np.cos(i)])
    e = np.radians(OBLIQ_J2000)
    return np.array([k[0], np.cos(e) * k[1] - np.sin(e) * k[2], np.sin(e) * k[1] + np.cos(e) * k[2]])


def precession(H):
    """
    Orbit-averaged solar precession of the spin pole (paper Eq. 1).

    dp/dt = K H (k . p) (p x k), K = 3 n^2 / (2 omega (1 - e^2)^{3/2}): the pole p circles the
    orbit normal k at the rate Omega_p = K H cos(eps).

    Parameters
    ----------
    H : float or ndarray
        Dynamic flattening.

    Returns
    -------
    dict
        eps (obliquity, deg), omega_p (rad/yr), period (My), dra, ddec (pole drift,
        mas/yr), rate (pole speed on the sky, mas/yr).
    """
    H = np.asarray(H, dtype=float)
    yr = 365.25 * 86400.0
    n2 = GM_SUN / (ORBIT["a"] * AU) ** 3
    omega = np.radians(SPIN_RATE) / 86400.0
    K = 1.5 * n2 / omega / (1.0 - ORBIT["e"] ** 2) ** 1.5 * yr  # [rad/yr]
    p, k = unit_radec(POLE_RA, POLE_DEC), orbit_normal_icrf(ORBIT["i"], ORBIT["node"])
    cos_eps = p @ k
    dp = np.multiply.outer(K * H * cos_eps, np.cross(p, k))  # [rad/yr]
    a, d = np.radians(POLE_RA), np.radians(POLE_DEC)
    mas = np.degrees(1.0) * 3.6e6
    dra = (-np.sin(a) * dp[..., 0] + np.cos(a) * dp[..., 1]) / np.cos(d) * mas
    ddec = dp[..., 2] / np.cos(d) * mas
    omega_p = K * H * cos_eps
    return dict(
        eps=np.degrees(np.arccos(cos_eps)), omega_p=omega_p, period=2 * np.pi / np.abs(omega_p) / 1e6,
        dra=dra, ddec=ddec, rate=np.linalg.norm(dp, axis=-1) * mas,
    )


# ============================================================
# Posteriors
# ============================================================


class MomentsPosterior:
    """
    MCMC (a): theta = (A_bar, B_bar, C_bar), data J2, C22 and optionally H.

    Prior: uniform on lo <= theta <= hi with A <= B <= C and A + B >= C (triangle inequality).

    Parameters
    ----------
    lo, hi : ndarray, shape (3,)
        Prior box.
    obs : dict
        Observed J2, C22, H.
    sigma : dict
        Their standard deviations; sigma["H"] = None drops H.
    """

    def __init__(self, lo, hi, obs, sigma):
        self.lo, self.hi = np.asarray(lo), np.asarray(hi)
        self.keys = [k for k in ("J2", "C22", "H") if sigma.get(k) is not None]
        self.y = np.array([obs[k] for k in self.keys])
        self.s = np.array([sigma[k] for k in self.keys])

    @staticmethod
    def predict(theta):
        a, b, c = theta
        J2 = c - 0.5 * (a + b)
        return dict(J2=J2, C22=0.25 * (b - a), H=J2 / c)

    def residuals(self, theta):
        p = self.predict(theta)
        return (np.array([p[k] for k in self.keys]) - self.y) / self.s

    def in_prior(self, theta):
        a, b, c = theta
        return bool(np.all(theta >= self.lo) and np.all(theta <= self.hi) and a <= b <= c and a + b >= c)

    def __call__(self, theta):
        if not self.in_prior(theta):
            return -np.inf
        r = self.residuals(theta)
        return -0.5 * r @ r

    def sample_prior(self, n, rng):
        out = []
        while sum(len(o) for o in out) < n:
            x = self.lo + (self.hi - self.lo) * rng.random((4 * n, 3))
            ok = (x[:, 0] <= x[:, 1]) & (x[:, 1] <= x[:, 2]) & (x[:, 0] + x[:, 1] >= x[:, 2])
            out.append(x[ok])
        return np.concatenate(out)[:n]


class TwoLayerPosterior:
    """
    MCMC (b): theta = (r2 [km], rho2 [kg/m^3], w), data J2, C22, H.

    Prior: uniform box PRIOR_B_LO..PRIOR_B_HI with rho1 >= RHO1_MIN.

    Parameters
    ----------
    model : TwoLayer
    obs, sigma : dict
        Observed J2, C22, H and their standard deviations.
    """

    keys = ("J2", "C22", "H")

    def __init__(self, model, obs, sigma, lo=PRIOR_B_LO, hi=PRIOR_B_HI):
        self.model, self.lo, self.hi = model, np.asarray(lo), np.asarray(hi)
        self.y = np.array([obs[k] for k in self.keys])
        self.s = np.array([sigma[k] for k in self.keys])

    def predict(self, theta):
        return self.model(*theta)

    def residuals(self, theta):
        d = self.predict(theta)
        if not np.isfinite(d["rho1"]):
            return np.full(len(self.keys), 1e6)
        return (np.array([d[k] for k in self.keys]) - self.y) / self.s

    def in_prior(self, theta):
        if np.any(theta < self.lo) or np.any(theta > self.hi):
            return False
        return self.model(*theta)["rho1"] >= RHO1_MIN

    def __call__(self, theta):
        if np.any(theta < self.lo) or np.any(theta > self.hi):
            return -np.inf
        d = self.predict(theta)
        if not d["rho1"] >= RHO1_MIN:
            return -np.inf
        r = (np.array([d[k] for k in self.keys]) - self.y) / self.s
        return -0.5 * r @ r

    def sample_prior(self, n, rng):
        out = []
        while len(out) < n:
            x = self.lo + (self.hi - self.lo) * rng.random(3)
            if self.in_prior(x):
                out.append(x)
        return np.array(out)

    def derived(self, theta):
        """DER_NAMES of one sample."""
        d = self.predict(theta)
        return np.array([d["I"], d["C"], d["H"], d["J2"], d["rho1"], d["r_c"]])


class LogCoreCoordinates:
    """
    Sampling coordinates for MCMC (b): phi = (ln r2, ln(rho2 - rho_mean), w).

    J2 and C22 fix the core excess mass moment, roughly (rho2 - rho_mean) r2^5, so in
    theta the posterior is a thin curved ridge on which affine-invariant moves stall
    (acceptance ~3 %, stuck walkers). In phi the ridge is close to a straight line.
    The log-Jacobian r2 (rho2 - rho_mean) keeps the prior uniform in theta, so the
    posterior is unchanged.

    Parameters
    ----------
    post : TwoLayerPosterior
    rho_mean : float
    """

    def __init__(self, post, rho_mean=RHO_MEAN):
        self.post, self.rho_mean = post, rho_mean

    def to_phi(self, theta):
        theta = np.asarray(theta, dtype=float)
        excess = np.maximum(theta[..., 1] - self.rho_mean, 1e-9)
        return np.stack([np.log(theta[..., 0]), np.log(excess), theta[..., 2]], axis=-1)

    def to_theta(self, phi):
        phi = np.asarray(phi, dtype=float)
        return np.stack([np.exp(phi[..., 0]), self.rho_mean + np.exp(phi[..., 1]), phi[..., 2]], axis=-1)

    @staticmethod
    def log_jacobian(phi):
        """ln |d theta / d phi| = ln r2 + ln(rho2 - rho_mean)."""
        return phi[..., 0] + phi[..., 1]

    def __call__(self, phi):
        lp = self.post(self.to_theta(phi))
        return lp + self.log_jacobian(phi) if np.isfinite(lp) else -np.inf


def initialize_walkers(post, n_walkers, rng, chi2_tol=25.0):
    """
    One bounded least-squares fit per walker from a prior draw, then a small Gaussian jitter.

    With fewer informative data than parameters the fits land at different points of the
    degenerate set, so the walkers start spread along it. Fits worse than the best by more
    than chi2_tol (local minima) are replaced by good ones.

    Returns
    -------
    p0 : ndarray, shape (n_walkers, ndim)
    theta_map : ndarray, shape (ndim,)
        Best fit.
    """
    width = post.hi - post.lo
    sols = []
    for x0 in post.sample_prior(n_walkers, rng):
        sol = least_squares(
            post.residuals, x0, bounds=(post.lo, post.hi), x_scale=width, method="trf",
            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=300,
        )
        if np.isfinite(post(sol.x)):
            sols.append(sol)
    if not sols:
        raise RuntimeError("no least-squares start reached the prior support")
    chi2 = np.array([2.0 * s.cost for s in sols])
    good = [s for s, c in zip(sols, chi2) if c <= chi2.min() + chi2_tol]
    p0 = []
    while len(p0) < n_walkers:
        s = good[len(p0) % len(good)]
        cov = np.linalg.pinv(s.jac.T @ s.jac) + np.diag((1e-5 * width) ** 2)
        th = s.x + np.linalg.cholesky(0.5 * (cov + cov.T)) @ rng.standard_normal(s.x.size) * 0.3
        if np.isfinite(post(th)):
            p0.append(th)
    best = good[int(np.argmin([g.cost for g in good]))]
    print(f"[Init] {len(sols)}/{n_walkers} fits in the prior, {len(good)} kept; "
          f"best chi2 = {2 * best.cost:.3f} ({best.fun.size} data, {best.x.size} parameters)")
    return np.array(p0), best.x


def run_emcee(post, n_steps, rng, label, coords=None):
    """
    emcee with differential-evolution moves (DEMove 0.8, DESnookerMove 0.2).

    Parameters
    ----------
    coords : LogCoreCoordinates, optional
        Sample in these coordinates instead of theta. Walkers start from the theta
        least-squares fits; the returned chain and log posterior are in theta.

    Returns
    -------
    chain : ndarray, shape (n_steps, n_walkers, ndim)
    logp : ndarray, shape (n_steps, n_walkers)
    tau : ndarray, shape (ndim,)
        Autocorrelation times in the sampled coordinates.
    theta_map : ndarray, shape (ndim,)
    """
    p0, theta_map = initialize_walkers(post, N_WALKERS, rng)
    target = post if coords is None else coords
    if coords is not None:
        p0 = coords.to_phi(p0)
    moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    sampler = emcee.EnsembleSampler(N_WALKERS, p0.shape[1], target, moves=moves)
    sampler.run_mcmc(p0, n_steps, progress=False)
    tau = sampler.get_autocorr_time(tol=0)
    print(f"[MCMC] {label}: acceptance {np.mean(sampler.acceptance_fraction):.3f}, "
          f"tau max {np.max(tau):.1f} ({n_steps / np.max(tau):.0f} tau)")
    chain, logp = sampler.get_chain(), sampler.get_log_prob()
    if coords is not None:
        logp = logp - coords.log_jacobian(chain)
        chain = coords.to_theta(chain)
    return chain, logp, tau, theta_map


def flat_posterior(chain, tau):
    """Burn-in (5 tau) and thinning (tau / 2) as in get_interior_inertia.py."""
    burn = int(5 * np.max(tau)) if BURN_IN is None else int(BURN_IN)
    thin = max(1, int(0.5 * np.min(tau))) if THIN is None else int(THIN)
    if burn >= chain.shape[0]:
        print(f"[MCMC] WARNING: burn-in {burn} >= {chain.shape[0]} steps; using half the chain")
        burn = chain.shape[0] // 2
    return chain[burn::thin].reshape(-1, chain.shape[-1]), burn, thin


def with_mean_moment(x):
    """Append I_bar = (A + B + C) / 3 to samples of (A_bar, B_bar, C_bar)."""
    return np.column_stack([x, x.mean(axis=1)])


# ============================================================
# Figures
# ============================================================


def plot_endmembers(r2s, rho2s, maps, r_min, extremes, name="fig_endmembers"):
    """
    Paper Fig. 2: H (black), J2 (vermillion) and rho1 (grey dashed) over core radius and
    density for the spherical (A) and rescaled-shape (B) cores; rho1 < RHO1_MIN shaded.
    """
    fig, axs = plt.subplots(1, 2, figsize=(MCMCPlots.TXT_W, 2.9), sharey=True, layout="constrained")
    X, Y = np.meshgrid(r2s, rho2s / 1e3, indexing="ij")
    for ax, m, title, ext in zip(axs, maps, ("(A) Spherical Core", "(B) Rescaled-Shape Core"), extremes):
        bad = np.where(np.isfinite(m["rho1"]) & (m["rho1"] < RHO1_MIN), 1.0, np.nan)
        ax.contourf(X, Y, bad, levels=[0.5, 1.5], colors=[MCMCPlots.C_SPAN], zorder=0)
        keep = m["rho1"] >= 0.9 * RHO1_MIN  # H, J2 only where the mantle is admissible (with margin)
        H, J2 = np.where(keep, m["H"], np.nan), np.where(keep, m["J2"], np.nan)
        if np.nanmax(H) - np.nanmin(H) > 1e-3:
            cs = ax.contour(X, Y, H, levels=np.arange(0.18, 0.30, 0.02), colors="k", linewidths=0.7)
            ax.clabel(cs, fontsize=6, fmt="%.2f", inline_spacing=2)
        else:  # rescaled-shape core: H does not depend on the core (paper Sec. 4.1)
            ax.text(0.04, 0.05, rf"$H = {np.nanmean(H):.4f}$ everywhere", transform=ax.transAxes,
                    fontsize=7, bbox=dict(fc="white", ec="none", pad=1.5))
        cj = ax.contour(X, Y, J2, levels=np.arange(0.06, 0.16, 0.02), colors=MCMCPlots.C_BATCH,
                        linewidths=0.7)
        ax.clabel(cj, fontsize=6, fmt="%.2f", inline_spacing=2)
        cr = ax.contour(X, Y, m["rho1"], levels=[2000, 3000, 4000], colors="0.55", linewidths=0.6,
                        linestyles="--")
        ax.clabel(cr, fontsize=5.5, fmt="%.0f", inline_spacing=2)
        if title.startswith("(A)"):
            ax.axvline(r_min, color="k", ls=":", lw=0.9)
            ax.text(r_min, 0.985, r"$r_{\min}$", transform=ax.get_xaxis_transform(), ha="right",
                    va="top", fontsize=7)
        (_, (i, j)) = ext
        ax.plot(r2s[i], rho2s[j] / 1e3, "*", ms=8, mfc=MCMCPlots.C_POST, mec="white", mew=0.5, zorder=5)
        ax.set_title(title)
        ax.set_xlabel(r"Core Radius $r_2$ [km]")
    axs[0].set_ylabel(r"Core Density $\rho_2$ [g/cm$^3$]")
    handles = [
        Line2D([], [], color="k", lw=0.7, label=r"$H$"),
        Line2D([], [], color=MCMCPlots.C_BATCH, lw=0.7, label=r"$J_2$"),
        Line2D([], [], color="0.55", lw=0.6, ls="--", label=r"$\rho_1$ [kg/m$^3$]"),
        Patch(color=MCMCPlots.C_SPAN, label=rf"$\rho_1 < {RHO1_MIN:.0f}$ kg/m$^3$"),
        Line2D([], [], ls="", marker="*", ms=8, mfc=MCMCPlots.C_POST, mec="white",
               label=r"Minimum $\bar{I}$ ($\rho_1 \geq %.0f$)" % RHO1_MIN),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=5, frameon=False)
    return MCMCPlots.save_figure(fig, name)


def plot_sigma_I(sweep, sig_I, sig_I_noH, C_true, C_hom, name="fig_sigma_I"):
    """
    Posterior standard deviation of I_bar against sigma_H / H (MCMC (a)), with Eq. (13)
    for the truth C_bar and the 75-110 % range of C_bar, and the paper's requirement.
    """
    x = np.linspace(0.0, 1.05 * max(sweep), 200)
    fig, ax = plt.subplots(figsize=(MCMCPlots.COL_W, 2.5), layout="constrained")
    ax.fill_between(100 * x, PRIOR_FRAC[0] * C_hom * x, PRIOR_FRAC[1] * C_hom * x,
                    color=MCMCPlots._tint(MCMCPlots.C_POST, 0.15), lw=0,
                    label=r"Eq. (13), $\bar{C}$ at 75--110$\%$ of Homogeneous")
    ax.plot(100 * x, C_true * x, color=MCMCPlots.C_POST, lw=0.9, label=r"Eq. (13), True $\bar{C}$")
    ax.plot(100 * np.array(sweep), sig_I, "o", ms=3.5, color=MCMCPlots.C_POST, label="MCMC (a)")
    ax.axhline(sig_I_noH, color=MCMCPlots.C_PRIOR, ls="--", lw=0.9, label=r"MCMC (a) without $H$")
    ax.axhline(0.01, color=MCMCPlots.C_TRUTH, ls=":", lw=0.8)
    for s in PAPER_SIGMA_H_REL:
        ax.plot(s, 0.01, "D", ms=3.5, mfc="white", mec=MCMCPlots.C_BATCH, mew=0.9)
    ax.plot([], [], "D", ms=3.5, mfc="white", mec=MCMCPlots.C_BATCH, label=r"Paper, $\sigma_{\bar{I}} = 0.01$")
    ax.set_xlabel(r"$\sigma_H / H$ [$\%$]")
    ax.set_ylabel(r"$\sigma_{\bar{I}}$ [-]")
    ax.set_xlim(0, 100 * x[-1])
    ax.set_ylim(0, 1.15 * max(np.max(sig_I), sig_I_noH))
    fig.legend(loc="outside upper center", ncol=2, frameon=False, fontsize=6.5)
    return MCMCPlots.save_figure(fig, name)


def plot_mean_moment(curves, I_true, I_hom, I_min, name="fig_mean_moment"):
    """
    Marginals of I_bar: prior (a), MCMC (a) without and with H, MCMC (b), against the truth,
    the homogeneous body and the paper's lower bound.

    Parameters
    ----------
    curves : list of (samples, label, color, linestyle, filled)
    """
    allx = np.concatenate([c[0] for c in curves])
    bins = np.linspace(*np.percentile(allx, [0.1, 99.9]), 90)
    fig, ax = plt.subplots(figsize=(MCMCPlots.COL_W, 2.4), layout="constrained")
    for x, lab, col, ls, filled in curves:
        h, e = np.histogram(x, bins=bins, density=True)
        if filled:
            ax.stairs(h, e, fill=True, color=MCMCPlots._tint(col, 0.35), lw=0)
        ax.stairs(h, e, color=col, ls=ls, lw=0.9, label=lab)
    ax.axvline(I_true, color=MCMCPlots.C_TRUTH, lw=1.0, label="Truth")
    ax.axvline(I_hom, color=MCMCPlots.C_BATCH, ls=":", lw=1.2, label="Homogeneous")
    ax.axvline(I_min, color="0.45", ls="-.", lw=0.9, label=r"Paper Minimum (Two-Layer)")
    ax.set_xlabel(r"$\bar{I} = (\bar{A} + \bar{B} + \bar{C}) / 3$ [-]")
    ax.set_ylabel("Density")
    ax.set_yticks([])
    fig.legend(loc="outside upper center", ncol=2, frameon=False, fontsize=6.5)
    return MCMCPlots.save_figure(fig, name)


def plot_traces(chain, logp, burn, truth, name="fig_traces"):
    """Walker traces of MCMC (b) and log posterior, burn-in shaded."""
    n_par = chain.shape[-1]
    fig, axs = plt.subplots(n_par + 1, 1, figsize=(MCMCPlots.TXT_W, 1.1 * (n_par + 1)), sharex=True,
                            layout="constrained")
    it = np.arange(chain.shape[0])
    labels = [MCMCPlots.param_label(i) for i in range(n_par)] + [r"$\log p$"]
    for i, ax in enumerate(axs):
        y = chain[..., i] if i < n_par else logp
        ax.axvspan(0, burn, color=MCMCPlots.C_SPAN, lw=0, zorder=0)
        ax.plot(it, y, color=MCMCPlots.C_POST, lw=0.3, alpha=0.25, rasterized=True)
        if i < n_par:
            ax.axhline(truth[i], color=MCMCPlots.C_TRUTH, ls="--", lw=0.8)
        else:
            ax.set_ylim(np.percentile(logp[burn:], 0.5) - 2, np.max(logp) + 1)
        ax.set_ylabel(labels[i])
    axs[-1].set_xlabel("Step")
    return MCMCPlots.save_figure(fig, name)


def plot_core_sections(C, S, model, post, truth, name="fig_core_sections", n_draw=80):
    """
    Outer shape and core boundaries in the x-z (meridional) and x-y (equatorial) planes of
    the shape frame: truth (dashed), posterior draws of MCMC (b) (blue), the r_min sphere.
    """
    t = np.linspace(0.0, 2.0 * np.pi, 721)
    planes = [("x", "z", np.column_stack([np.cos(t), np.zeros_like(t), np.sin(t)]), (0, 2)),
              ("x", "y", np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)]), (0, 1))]
    fig, axs = plt.subplots(1, 2, figsize=(MCMCPlots.TXT_W, 2.9), layout="constrained")
    r_min = model.g.R.min()
    for ax, (lx, ly, u, (i, j)) in zip(axs, planes):
        R = sh_radius(C, S, u)
        ax.fill(R * u[:, i] / 1e3, R * u[:, j] / 1e3, color=MCMCPlots._tint("0.5", 0.18), lw=0)
        ax.plot(R * u[:, i] / 1e3, R * u[:, j] / 1e3, color="0.25", lw=0.9)
        for th in post[:n_draw]:
            rc = model.core_radius(1e3 * th[0], th[2], R)
            ax.plot(rc * u[:, i] / 1e3, rc * u[:, j] / 1e3, color=MCMCPlots.C_POST, lw=0.4, alpha=0.2)
        rc = model.core_radius(1e3 * truth[0], truth[2], R)
        ax.plot(rc * u[:, i] / 1e3, rc * u[:, j] / 1e3, color=MCMCPlots.C_TRUTH, lw=1.0, ls="--")
        ax.plot(r_min * np.cos(t) / 1e3, r_min * np.sin(t) / 1e3, color="0.45", lw=0.7, ls=":")
        ax.set_aspect("equal")
        ax.set_xlabel(f"${lx}$ [km]")
        ax.set_ylabel(f"${ly}$ [km]")
    handles = [
        Line2D([], [], color="0.25", lw=0.9, label="Surface (Shepard et al. 2021)"),
        Line2D([], [], color=MCMCPlots.C_TRUTH, ls="--", lw=1.0, label="Core (Truth)"),
        Line2D([], [], color=MCMCPlots.C_POST, lw=0.8, label="Core (Posterior Draws)"),
        Line2D([], [], color="0.45", ls=":", lw=0.7, label=r"$r_{\min}$ Sphere"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False)
    return MCMCPlots.save_figure(fig, name)


# ============================================================
# Summary
# ============================================================


def _row(name, paper, ours, fmt="{:.4f}"):
    p = paper if isinstance(paper, str) else fmt.format(paper)
    o = ours if isinstance(ours, str) else fmt.format(ours)
    return f"  {name:<46s} {p:>18s} {o:>22s}"


def _ci(x):
    lo, med, hi = np.percentile(x, [16, 50, 84])
    return f"{med:.4f} +{hi - med:.4f} -{med - lo:.4f}"


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    t_start = time.time()
    MCMCPlots.configure(SYM_B, UNIT_B, np.ones(3), body=BODY)

    # --- 1. shape and homogeneous body ---
    Csh, Ssh = load_sh_shape(SHAPE_SH)
    grid = AngularGrid(Csh, Ssh, N_MU, N_PHI)
    V = grid.outer[0]
    r_min = grid.R.min() / 1e3
    hom = inertia_quantities(*(RHO_MEAN * np.asarray(x) for x in grid.outer))
    print(f"[Shape] {SHAPE_SH.name}: degree {Csh.shape[0] - 1}, {grid.w.size} nodes, "
          f"V = {V:.5e} m^3, R_v = {grid.r_v / 1e3:.3f} km, r_min = {r_min:.2f} km, "
          f"r_max = {grid.R.max() / 1e3:.2f} km, CoM offset {np.linalg.norm(hom['com']) / 1e3:.2f} km")
    print("[Homogeneous] " + ", ".join(f"{k} = {hom[k]:.4f} (paper {PAPER_HOM[k]:.4f})" for k in PAPER_HOM)
          + f", I = {hom['I']:.4f}")
    model = TwoLayer(grid, RHO_MEAN)
    t0 = time.time()
    for _ in range(200):
        model(*TRUTH)
    print(f"[Model] two-layer moments: {(time.time() - t0) / 200 * 1e3:.3f} ms per call")

    # --- 2. endmember maps ---
    r2s, rho2s = np.linspace(*GRID_R2[:2], GRID_R2[2]), np.linspace(*GRID_RHO2[:2], GRID_RHO2[2])
    maps = [endmember_map(model, w, r2s, rho2s) for w in (0.0, 1.0)]
    ok_A = maps[0]["rho1"] >= RHO1_MIN
    minH_A, _ = masked_extreme(maps[0]["H"], ok_A, np.nanargmin)
    minI_A, kI_A = masked_extreme(maps[0]["I"], ok_A, np.nanargmin)
    minI_A8, _ = masked_extreme(maps[0]["I"], ok_A & (rho2s[None, :] <= RHO2_METAL), np.nanargmin)
    ok_B = maps[1]["rho1"] >= RHO1_MIN
    minI_B, kI_B = masked_extreme(maps[1]["I"], ok_B, np.nanargmin)
    dH_B = np.nanmax(np.abs(np.where(ok_B, maps[1]["H"], np.nan) - hom["H"]))
    minJ2_B, _ = masked_extreme(maps[1]["J2"], ok_B, np.nanargmin)
    H_grid = moment_grid_H_range(hom)
    print(f"[Endmember A] rho1 >= {RHO1_MIN:.0f}: min H = {minH_A:.4f} (paper {PAPER_MIN_H}), "
          f"min I = {minI_A:.4f} (paper {PAPER_MIN_I}; {minI_A8:.4f} with rho2 <= {RHO2_METAL:.0f})")
    print(f"[Endmember B] max |H - H_hom| = {dH_B:.1e}, min J2 = {minJ2_B:.4f}, min I = {minI_B:.4f}")
    print(f"[Grid 75-110 %] H in [{H_grid[0]:.3f}, {H_grid[1]:.3f}] (paper {PAPER_H_RANGE})")

    # --- 5. precession of the homogeneous body ---
    prec = precession(hom["H"])
    # the period of Eq. (1) scales with 1 / |cos(eps)|; the obliquity oscillates over the cycle,
    # so the period of the full cycle follows from the mean of the paper's 10-My range
    eps_mean = np.mean(PAPER_OBLIQUITY)
    period_mean = prec["period"] * np.cos(np.radians(prec["eps"])) / np.cos(np.radians(eps_mean))
    print(f"[Precession] obliquity {prec['eps']:.2f} deg, period {prec['period']:.2f} My "
          f"({period_mean:.2f} My at the mean obliquity {eps_mean:.1f} deg; paper {PAPER_PREC_PERIOD}), "
          f"d alpha/dt = {prec['dra']:.1f}, d delta/dt = "
          f"{prec['ddec']:.1f} mas/yr (paper {PAPER_POLE_RATE})")

    # --- synthetic data ---
    d_true = model(*TRUTH)
    rng_d = np.random.default_rng(SEED_DATA)
    z = rng_d.standard_normal(3)  # one noise realization, rescaled for every sigma
    sig_J2 = SIGMA_J2_REL * d_true["J2"]
    sig_C22 = SIGMA_J2_REL * d_true["C22"]
    print(f"[Truth] r2 = {TRUTH[0]} km, rho2 = {TRUTH[1]:.0f}, w = {TRUTH[2]}: rho1 = {d_true['rho1']:.1f}, "
          f"r_c = {d_true['r_c']:.2f} km, A, B, C = {d_true['A']:.4f}, {d_true['B']:.4f}, "
          f"{d_true['C']:.4f}, J2 = {d_true['J2']:.4f}, C22 = {d_true['C22']:.5f}, H = {d_true['H']:.4f}, "
          f"I = {d_true['I']:.4f}")

    def observe(sig_H_rel, floor=0.0):
        """Data J2, C22, H and sigmas for one sigma_H / H (shared noise draws z)."""
        sJ2 = np.hypot(sig_J2, floor * d_true["J2"])
        sC22 = np.hypot(sig_C22, floor * d_true["C22"])
        sH = None if sig_H_rel is None else sig_H_rel * d_true["H"]
        obs = dict(J2=d_true["J2"] + sJ2 * z[0], C22=d_true["C22"] + sC22 * z[1],
                   H=d_true["H"] + (0.0 if sH is None else sH * z[2]))
        return obs, dict(J2=sJ2, C22=sC22, H=sH)

    lo_A = PRIOR_FRAC[0] * np.array([hom["A"], hom["B"], hom["C"]])
    hi_A = PRIOR_FRAC[1] * np.array([hom["A"], hom["B"], hom["C"]])
    theta_A_true = np.array([d_true["A"], d_true["B"], d_true["C"]])
    obs_B, sig_B = observe(SIGMA_H_REL, SIGMA_MODEL_REL)
    post_B_model = TwoLayerPosterior(model, obs_B, sig_B)

    fig_dir = MCMCPlots.set_figure_dir(
        HERE / "results" / (Path(__file__).stem + "_" + datetime.now().strftime("%y%m%d%H%M%S"))
    )

    # --- 3-4. sampling ---
    sweep = list(SIGMA_H_SWEEP) + [None]  # None: no H
    rng = np.random.default_rng()
    if LOAD_CHAIN:
        saved = np.load(LOAD_CHAIN, allow_pickle=True)
        chains_A = list(saved["chains_A"])
        taus_A = list(saved["taus_A"])
        chain_B, logp_B, tau_B = saved["chain_B"].astype(float), saved["logp_B"].astype(float), saved["tau_B"]
        print(f"[MCMC] loaded {LOAD_CHAIN}")
    else:
        chains_A, taus_A = [], []
        for s in sweep:
            obs, sig = observe(s)
            ch, _, tau, _ = run_emcee(MomentsPosterior(lo_A, hi_A, obs, sig), N_STEPS_A, rng,
                                      f"(a) sigma_H/H = {'none' if s is None else f'{100 * s:.1f} %'}")
            chains_A.append(ch.astype(np.float32))
            taus_A.append(tau)
        chain_B, logp_B, tau_B, map_B = run_emcee(post_B_model, N_STEPS_B, rng, "(b) two-layer",
                                                   coords=LogCoreCoordinates(post_B_model))
        buf = io.BytesIO()
        np.savez_compressed(
            buf, chains_A=np.array(chains_A, dtype=object), taus_A=np.array(taus_A, dtype=object),
            sweep=np.array(sweep, dtype=object), chain_B=chain_B.astype(np.float32),
            logp_B=logp_B.astype(np.float32), tau_B=tau_B, map_B=map_B, truth=TRUTH,
            obs_B=np.array(obs_B, dtype=object), sig_B=np.array(sig_B, dtype=object), noise=z,
            prior_A=np.array([lo_A, hi_A]), prior_B=np.array([PRIOR_B_LO, PRIOR_B_HI]),
            shape=SHAPE_SH.name, n_mu=N_MU, n_phi=N_PHI, rho_mean=RHO_MEAN, rho1_min=RHO1_MIN,
        )
        MCMCPlots.write_output(fig_dir / "chain.npz", buf.getvalue(), tag="Chain")

    # --- posteriors (a) ---
    flat_A = [with_mean_moment(flat_posterior(ch.astype(float), tau)[0]) for ch, tau in zip(chains_A, taus_A)]
    sig_I = np.array([np.std(x[:, 3]) for x in flat_A[:-1]])
    sig_I_noH = np.std(flat_A[-1][:, 3])
    k18 = int(np.argmin(np.abs(np.array(SIGMA_H_SWEEP) - SIGMA_H_REL)))
    for s, x in zip(sweep, flat_A):
        lab = "none" if s is None else f"{100 * s:4.1f} %"
        eq13 = "" if s is None else f", Eq. 13: {d_true['C'] * s:.4f}"
        print(f"[Posterior a] sigma_H/H {lab}: I = {_ci(x[:, 3])}, sigma_I = {np.std(x[:, 3]):.4f}{eq13}")
    sigH_needed = 0.01 / np.array([PRIOR_FRAC[0] * hom["C"], hom["C"], PRIOR_FRAC[1] * hom["C"], d_true["C"]])
    sigH_mcmc = np.interp(0.01, sig_I, SIGMA_H_SWEEP)
    prior_A = with_mean_moment(MomentsPosterior(lo_A, hi_A, *observe(None)).sample_prior(N_PRIOR,
                                                                                     np.random.default_rng(4)))

    # --- posterior (b) ---
    post_B, burn_B, thin_B = flat_posterior(chain_B, tau_B)
    sub = post_B[np.random.default_rng(3).choice(len(post_B), min(N_DERIVED, len(post_B)), replace=False)]
    D_post = np.array([post_B_model.derived(th) for th in sub])
    prior_B = post_B_model.sample_prior(N_PRIOR // 4, np.random.default_rng(5))
    D_prior = np.array([post_B_model.derived(th) for th in prior_B])
    D_true = post_B_model.derived(TRUTH)
    print(f"[Posterior b] {len(post_B)} samples (burn-in {burn_B}, thin {thin_B}); "
          + ", ".join(f"{n} = {_ci(D_post[:, i])}" for i, n in enumerate(DER_NAMES[:4])))
    prec_B = precession(D_post[:, 2])

    # --- figures ---
    plot_endmembers(r2s, rho2s, maps, r_min, [(minI_A, kI_A), (minI_B, kI_B)])
    plot_sigma_I(SIGMA_H_SWEEP, sig_I, sig_I_noH, d_true["C"], hom["C"])
    plot_mean_moment(
        [
            (prior_A[:, 3], "Prior (a)", MCMCPlots.C_PRIOR, "--", False),
            (flat_A[-1][:, 3], r"MCMC (a), $J_2$, $C_{22}$", "0.45", "-", False),
            (flat_A[k18][:, 3], r"MCMC (a), $+ H$ (%.1f$\%%$)" % (100 * SIGMA_H_REL), MCMCPlots.C_POST, "-", True),
            (D_post[:, 0], "MCMC (b), Two-Layer", MCMCPlots.C_SC, "-", False),
        ],
        d_true["I"], hom["I"], PAPER_MIN_I,
    )
    MCMCPlots.configure(SYM_A, ["", "", "", ""], np.ones(4))
    a_true = np.append(theta_A_true, d_true["I"])
    MCMCPlots.plot_corner(flat_A[k18], true_theta=a_true, prior=prior_A, prior_label="Prior",
                          name="fig_corner_moments_prior")
    MCMCPlots.plot_corner(flat_A[-1], true_theta=a_true, prior=prior_A, prior_label="Prior",
                          name="fig_corner_moments_noH_prior")
    MCMCPlots.configure(SYM_B, UNIT_B, np.ones(3))
    plot_traces(chain_B, logp_B, burn_B, TRUTH)
    MCMCPlots.plot_corner(post_B, true_theta=TRUTH, name="fig_corner")
    MCMCPlots.plot_corner(post_B, true_theta=TRUTH, prior=prior_B, prior_label="Prior",
                          name="fig_corner_prior")
    MCMCPlots.configure(DER_SYM, DER_UNIT, np.ones(len(DER_SYM)))
    MCMCPlots.plot_corner(D_post, true_theta=D_true, name="fig_corner_derived")
    MCMCPlots.plot_corner(D_post, true_theta=D_true, prior=D_prior, prior_label="Prior",
                          name="fig_corner_derived_prior")
    MCMCPlots.configure(SYM_B, UNIT_B, np.ones(3))
    plot_core_sections(Csh, Ssh, model, sub, TRUTH)

    # --- summary: paper vs this script ---
    head = f"  {'Quantity':<46s} {'Ermakov (2026)':>18s} {'This script':>22s}"
    lines = [
        f"16 Psyche moments of inertia: Ermakov (2026, PSJ 7:170) vs {Path(__file__).name}",
        f"Shape {SHAPE_SH.name} (Zenodo 10.5281/zenodo.12522382), degree {Csh.shape[0] - 1}; "
        f"quadrature {N_MU} x {N_PHI}; R0 = {R0 / 1e3:.0f} km; mean density {RHO_MEAN:.0f} kg/m^3",
        f"V = {V:.5e} m^3, R_v = {grid.r_v / 1e3:.3f} km, centre-of-mass offset "
        f"{np.linalg.norm(hom['com']) / 1e3:.3f} km",
        "",
        "1. Homogeneous body (paper Table 1, Shepard et al. 2021 row)",
        head,
    ]
    for k in PAPER_HOM:
        lines.append(_row(f"{k}" + ("" if k in ("J2", "C22", "H") else "_bar"), PAPER_HOM[k],
                          f"{hom[k]:.4f} ({100 * (hom[k] / PAPER_HOM[k] - 1):+.2f} %)"))
    I_paper = PAPER_HOM["J2"] * (1 / PAPER_HOM["H"] - 2 / 3)
    lines += [
        _row("I_bar = J2 (1/H - 2/3)", I_paper, hom["I"]),
        "",
        "2. Two-layer endmembers (paper Sec. 4.1, Fig. 2)",
        head,
        _row("r_min, core exposure onset [km]", PAPER_RMIN, r_min, "{:.1f}"),
        _row(f"(A) min H, rho1 >= {RHO1_MIN:.0f}", PAPER_MIN_H, minH_A),
        _row(f"(A) min I_bar, rho2 <= {GRID_RHO2[1]:.0f}", PAPER_MIN_I, minI_A),
        _row(f"(A) min I_bar, rho2 <= {RHO2_METAL:.0f}", "-", minI_A8),
        _row(f"(A) min I_bar at r2 [km], rho2 [kg/m^3]", "-",
             f"{r2s[kI_A[0]]:.1f}, {rho2s[kI_A[1]]:.0f}"),
        _row("(B) H change vs homogeneous", "unaffected", f"max {dH_B:.1e}"),
        _row("(B) J2", "< homogeneous", f">= {minJ2_B:.4f}"),
        _row("H range, A, B, C at 75-110 % (C largest)", f"{PAPER_H_RANGE[0]:.2f} - {PAPER_H_RANGE[1]:.2f}",
             f"{H_grid[0]:.3f} - {H_grid[1]:.3f}"),
        "",
        "3. MCMC (a): theta = (A_bar, B_bar, C_bar), prior 75-110 % of homogeneous",
        f"   data J2, C22 (sigma/value {100 * SIGMA_J2_REL:.3f} %) + H; truth = two-layer TRUTH; "
        f"{N_WALKERS} walkers x {N_STEPS_A} steps",
        f"  {'sigma_H/H':<12s} {'I_bar (16-50-84 %)':>30s} {'sigma_I MCMC':>14s} {'Eq. 13':>10s}",
    ]
    for s, x in zip(sweep, flat_A):
        lab = "no H" if s is None else f"{100 * s:.1f} %"
        eq = "-" if s is None else f"{d_true['C'] * s:.4f}"
        lines.append(f"  {lab:<12s} {_ci(x[:, 3]):>30s} {np.std(x[:, 3]):>14.4f} {eq:>10s}")
    lines += [
        f"  prior only: I_bar {_ci(prior_A[:, 3])}, sigma {np.std(prior_A[:, 3]):.4f}",
        head,
        _row("sigma_H/H for sigma_I = 0.01, C_bar 75 % hom", f"{PAPER_SIGMA_H_REL[1]:.1f} %",
             f"{100 * sigH_needed[0]:.2f} % (Eq. 13)", "{}"),
        _row("sigma_H/H for sigma_I = 0.01, C_bar homogeneous", f"{PAPER_SIGMA_H_REL[0]:.1f} %",
             f"{100 * sigH_needed[1]:.2f} % (Eq. 13)", "{}"),
        _row("sigma_H/H for sigma_I = 0.01, C_bar 110 % hom", "-", f"{100 * sigH_needed[2]:.2f} % (Eq. 13)", "{}"),
        _row("sigma_H/H for sigma_I = 0.01, truth C_bar", "-",
             f"{100 * sigH_needed[3]:.2f} % / {100 * sigH_mcmc:.2f} % MCMC", "{}"),
        "",
        "4. MCMC (b): theta = (r2 [km], rho2 [kg/m^3], w); prior "
        f"{PRIOR_B_LO.tolist()} - {PRIOR_B_HI.tolist()}, rho1 >= {RHO1_MIN:.0f}",
        f"   data J2, C22 (sigma {100 * sig_B['J2'] / d_true['J2']:.3f} %, model floor "
        f"{100 * SIGMA_MODEL_REL:.2f} %), H (sigma {100 * SIGMA_H_REL:.1f} %); {N_WALKERS} walkers x "
        f"{chain_B.shape[0]} steps in (ln r2, ln(rho2 - rho_mean), w), burn-in {burn_B}, thin {thin_B}, "
        f"tau max {np.max(tau_B):.1f}",
        f"  {'':<10s} {'truth':>12s} {'posterior (16-50-84 %)':>34s} {'prior (16-50-84 %)':>34s}",
    ]
    for i, n in enumerate(["r2", "rho2", "w"]):
        lines.append(f"  {n:<10s} {TRUTH[i]:>12.4f} {_ci(post_B[:, i]):>34s} {_ci(prior_B[:, i]):>34s}")
    for i, n in enumerate(DER_NAMES):
        lines.append(f"  {n:<10s} {D_true[i]:>12.4f} {_ci(D_post[:, i]):>34s} {_ci(D_prior[:, i]):>34s}")
    lines += [
        "",
        "5. Precession (paper Eq. 1, Table 3), homogeneous body",
        f"   pole RA {POLE_RA}, Dec {POLE_DEC} deg; orbit a {ORBIT['a']} AU, e {ORBIT['e']}, "
        f"i {ORBIT['i']} deg, node {ORBIT['node']} deg (approximate)",
        head,
        _row("obliquity [deg]", f"{PAPER_OBLIQUITY[0]} - {PAPER_OBLIQUITY[1]} (10 My)", prec["eps"], "{:.2f}"),
        _row("precession period, present obliquity [My]", "-", prec["period"], "{:.2f}"),
        _row(f"precession period, mean obliquity {eps_mean:.1f} deg [My]", PAPER_PREC_PERIOD,
             period_mean, "{:.2f}"),
        _row("d alpha / dt [mas/yr]", PAPER_POLE_RATE[0], prec["dra"], "{:.1f}"),
        _row("d delta / dt [mas/yr]", PAPER_POLE_RATE[1], prec["ddec"], "{:.1f}"),
        _row("pole drift rate, MCMC (b) H [mas/yr]", "-",
             f"{np.median(prec_B['rate']):.1f} +/- {np.std(prec_B['rate']):.1f}", "{}"),
        f"   pole drift rate is proportional to H (paper Eq. 12): sigma_H / H = "
        f"{100 * SIGMA_H_REL:.1f} % <-> {SIGMA_H_REL * prec['rate']:.1f} mas/yr on "
        f"{prec['rate']:.1f} mas/yr",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    MCMCPlots.write_output(fig_dir / "summary.txt", text.encode(), tag="Summary")
    print(f"[Done] {time.time() - t_start:.0f} s; outputs in {fig_dir}")
