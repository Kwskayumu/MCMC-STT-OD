r"""
Moments of inertia of 269 Justitia (EMA target) from degree-2 gravity and rotational
dynamics, by MCMC: the analysis of get_interior_inertia_psyche.py (Ermakov 2026, PSJ 7:170)
applied to Justitia, as a mission-design study with synthetic data.

Standalone: no orbit propagation, no MCMCModel. Only the figures reuse the house style of
MCMCPlots.py (configure, plot_corner, save_figure, write_output).

Shape
-----
Triangular mesh data/kernels_justitia/justitia_shape.obj (574 vertices, 1144 faces, km;
body frame with z along the spin axis of justitia_IAU.tpc). The mesh is star-shaped about
its centroid, so the surface radius R(u) along any direction follows from one ray-triangle
intersection (Moller-Trumbore), and every mass integral of a region {r < R(u)} is an angular
integral with an analytic radial part (unit density):

    mass = int R^3 / 3 dOmega,  first = int u R^4 / 4 dOmega,  second = int u u^T R^5 / 5 dOmega,

evaluated by Gauss-Legendre quadrature in cos(theta) times a uniform rule in phi. The
quadrature of the whole body is checked against the exact polyhedron integrals (signed
tetrahedra). Normalization: A_bar = A / (M R0^2), R0 = 25 km (volume-equivalent radius), and

    J2 = C_bar - (A_bar + B_bar) / 2,   C22 = (B_bar - A_bar) / 4,   H = J2 / C_bar,
    I_bar = (A_bar + B_bar + C_bar) / 3 = J2 (1/H - 2/3).

Justitia has no mass estimate: RHO_MEAN is an assumption (primitive, very red body). The
normalized moments of a two-layer body depend on the densities only through rho2 / rho_mean.

Steps (summary.txt)
-------------------
1. Homogeneous body: A_bar, B_bar, C_bar, J2, C22, H, I_bar (quadrature vs exact
   polyhedron), and the angle between the spin axis (body z) and the axis of maximum inertia.
2. Two-layer endmembers (as Ermakov 2026, Fig. 2): core of radius r2 and density rho2;
   mantle density rho1 from the mean density. (A) spherical core truncated by the surface
   where exposed (r2 > r_min), (B) core = outer shape rescaled to the volume-equivalent
   radius r2. Minimum H and I_bar with rho1 >= RHO1_MIN; H unchanged for (B).
3. MCMC (a), model-agnostic: theta = (A_bar, B_bar, C_bar), uniform prior on 75-110 % of the
   homogeneous values (A <= B <= C, A + B >= C); data J2, C22 (sigma/value SIGMA_J2_REL) and
   H from the spin-pole precession, for a sweep of sigma_H / H. The posterior sigma of I_bar
   is compared with Eq. (13), sigma_I = C_bar sigma_H / H; the sigma_H / H giving
   sigma_I = 0.01 is the requirement on H.
4. MCMC (b), physical two-layer model: theta = (r2, rho2, w), with the core boundary
   R_c(u) = min(R(u), r2 [(1 - w) + w R(u) / R_v]) blending endmember (A) (w = 0) into
   endmember (B) (w = 1); data J2, C22 (with a model-error floor) and H. Sampled in
   phi = (ln r2, ln(rho2 - rho_mean), w), which straightens the J2/C22 ridge
   (rho2 - rho_mean) r2^5 ~ const; the Jacobian keeps the prior uniform in theta.
5. Precession (Ermakov 2026, Eq. 1): obliquity, precession period and pole drift rate of
   the homogeneous body from the IAU spin pole and the heliocentric orbit; the drift rate
   scales with H, so sigma_H / H sets the pole-rate accuracy a mission must reach.

The synthetic data are the TRUTH two-layer model plus Gaussian noise (seed SEED_DATA).

Outputs (results/<script name>_<YYMMDDHHMMSS>/)
-----------------------------------------------
chain.npz (every chain, data and settings; replot with LOAD_CHAIN), summary.txt, and one
PDF per figure: endmember maps, sigma_I vs sigma_H / H, mean moment of inertia marginals,
moment corner (a), traces and corners (b), core cross-sections.
"""

from __future__ import annotations

import io
import time
from datetime import datetime
from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.optimize import least_squares

import MCMCPlots

MCMCPlots.SHOW_FIGS = True  # plt.show() after each figure is saved

HERE = Path(__file__).resolve().parent

# ============================================================
# Settings
# ============================================================

# --- Shape and bulk properties ---
SHAPE_OBJ = HERE / "data" / "kernels_justitia" / "justitia_shape.obj"  # triangular mesh [km]
BODY = "Justitia"
R0 = 25.0e3  # normalization radius [m] (volume-equivalent radius 25.01 km)
RHO_MEAN = 1500.0  # mean density [kg/m^3]: ASSUMED (no mass estimate of Justitia)
RHO1_MIN = 500.0  # minimum mantle density [kg/m^3] (porous, ice-rich regolith)
N_MU, N_PHI = 64, 128  # angular quadrature: Gauss-Legendre nodes in cos(theta) x uniform phi

# --- Endmember maps ---
GRID_R2 = (1.0, 25.0, 97)  # core radius r2 [km]: start, stop, count
GRID_RHO2 = (RHO_MEAN, 5000.0, 80)  # core density rho2 [kg/m^3]: start, stop, count
RHO2_ROCK = 3500.0  # anhydrous rock: second bound reported for the minimum I_bar, prior bound

# --- Rotation and orbit (precession) ---
POLE_RA, POLE_DEC = 85.069051546534496, -57.852303738160167  # spin pole, ICRF [deg] (justitia_IAU.tpc)
SPIN_RATE = 260.797885  # W rate [deg/day] (justitia_IAU.tpc; P = 33.13 h)
ORBIT = dict(a=2.6145, e=0.2164, i=5.4778, node=156.4409)  # [AU, -, deg, deg] ecliptic J2000
#   osculating elements on 2034-01-01 from Justitia_20240612.bsp (SPICE oscltx, ECLIPJ2000)
GM_SUN = 1.32712440018e20  # [m^3/s^2]
AU = 1.495978707e11  # [m]
OBLIQ_J2000 = 23.4392911  # obliquity of the ecliptic [deg]

# --- Truth (synthetic data): two-layer model, theta = (r2 [km], rho2 [kg/m^3], w) ---
TRUTH = np.array([15.0, 2500.0, 0.5])

# --- Measurement noise (assumed for a rendezvous radio-science campaign) ---
SEED_DATA = 2026  # noise realization of the synthetic data
SIGMA_J2_REL = 1.0e-3  # sigma / value for J2 and C22
SIGMA_H_REL = 0.02  # sigma_H / H of MCMC (b)
SIGMA_H_SWEEP = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]  # sigma_H / H of the MCMC (a) runs
SIGMA_MODEL_REL = 1.0e-3  # MCMC (b): model-error floor on J2, C22 (shape, quadrature)

# --- Priors ---
PRIOR_FRAC = (0.75, 1.10)  # MCMC (a): each moment within this fraction of homogeneous
PRIOR_B_LO = np.array([2.0, RHO_MEAN, 0.0])  # MCMC (b): r2 [km], rho2 [kg/m^3], w
PRIOR_B_HI = np.array([25.0, RHO2_ROCK, 1.0])

# --- MCMC ---
N_WALKERS = 32
N_STEPS_A = 6000  # per sigma_H / H of the sweep (and the run without H)
N_STEPS_B = 30_000
BURN_IN = None  # None: 5 x max autocorrelation time
THIN = None  # None: max(1, min autocorrelation time / 2)
LOAD_CHAIN = None  # e.g. "results/get_interior_inertia_justitia_<stamp>/chain.npz" (replot only)

N_DERIVED = 4000  # posterior samples of (b) mapped to derived quantities
N_PRIOR = 20_000  # prior samples for the prior overlays

SYM_A = [r"\bar{A}", r"\bar{B}", r"\bar{C}", r"\bar{I}"]
SYM_B = [r"r_2", r"\rho_2", r"w"]
UNIT_B = ["km", "kg/m$^3$", ""]
DER_NAMES = ["I_bar", "C_bar", "H", "J2", "rho1", "r_c"]
DER_SYM = [r"\bar{I}", r"\bar{C}", r"H", r"J_2", r"\rho_1", r"r_{\mathrm{c}}"]
DER_UNIT = ["", "", "", "", "kg/m$^3$", "km"]


# ============================================================
# Shape model (triangular mesh) and mass integrals
# ============================================================


def load_obj(path):
    """
    Read a triangular OBJ mesh.

    Parameters
    ----------
    path : str or Path
        Wavefront OBJ with "v x y z" [km] and "f i j k" (1-based, outward-oriented) lines.

    Returns
    -------
    V : ndarray, shape (n_vertices, 3)
        Vertices [m].
    F : ndarray, shape (n_faces, 3)
        Vertex indices of each face (0-based).
    """
    V, F = [], []
    with open(path) as fh:
        for line in fh:
            t = line.split()
            if t and t[0] == "v":
                V.append([float(x) for x in t[1:4]])
            elif t and t[0] == "f":
                F.append([int(x.split("/")[0]) - 1 for x in t[1:4]])
    return 1e3 * np.array(V), np.array(F)


def polyhedron_integrals(V, F):
    """
    Exact unit-density mass integrals of a closed polyhedron (signed tetrahedra from the origin).

    Returns
    -------
    vol : float
        [m^3].
    first : ndarray, shape (3,)
        int x dV [m^4].
    second : ndarray, shape (3, 3)
        int x x^T dV [m^5].
    """
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    s = a + b + c
    second = sum(np.einsum("n,ni,nj->ij", det, x, x) for x in (a, b, c, s)) / 120.0
    return det.sum() / 6.0, det @ s / 24.0, second


def mesh_radius(V, F, u, chunk=1024):
    """
    Surface radius of a star-shaped mesh along unit directions from the origin.

    Moller-Trumbore ray-triangle intersection of every ray with every face; the nearest hit
    in front of the origin is the surface.

    Parameters
    ----------
    V, F : ndarray
        Mesh (load_obj), star-shaped about the origin.
    u : ndarray, shape (N, 3)
        Unit directions.

    Returns
    -------
    ndarray, shape (N,)
        R(u) [m].
    """
    a = V[F[:, 0]]
    e1, e2 = V[F[:, 1]] - a, V[F[:, 2]] - a
    q = np.cross(-a, e1)  # ray origin at 0: s = -a
    t_num = np.einsum("mi,mi->m", e2, q)
    R = np.empty(len(u))
    tol = 1e-9
    for k in range(0, len(u), chunk):
        d = u[k:k + chunk]
        p = np.cross(d[:, None, :], e2[None])
        det = np.einsum("nmi,mi->nm", p, e1)
        with np.errstate(divide="ignore", invalid="ignore"):
            bu = np.einsum("mi,nmi->nm", -a, p) / det
            bv = (d @ q.T) / det
            t = t_num[None] / det
            hit = (bu >= -tol) & (bv >= -tol) & (bu + bv <= 1 + tol) & (t > 0)
        R[k:k + chunk] = np.where(hit, t, np.inf).min(axis=1)
    if not np.all(np.isfinite(R)):
        raise ValueError("some rays miss the mesh: not star-shaped about the origin")
    return R


class AngularGrid:
    """
    Angular quadrature of a star-shaped body and its unit-density mass integrals.

    Parameters
    ----------
    V, F : ndarray
        Mesh (load_obj), recentred on its centroid.
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

    def __init__(self, V, F, n_mu, n_phi):
        mu, wmu = np.polynomial.legendre.leggauss(n_mu)
        phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
        MU, PHI = (a.ravel() for a in np.meshgrid(mu, phi, indexing="ij"))
        st = np.sqrt(1.0 - MU**2)
        self.u = np.column_stack([st * np.cos(PHI), st * np.sin(PHI), MU])
        self.w = np.outer(wmu, np.full(n_phi, 2.0 * np.pi / n_phi)).ravel()
        self._uu = np.einsum("ni,nj->nij", self.u, self.u).reshape(-1, 9)
        self.R = mesh_radius(V, F, self.u)
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
        A, B, C (normalized by M R0^2, ascending), J2, C22, H, I (mean moment), com [m],
        axes (principal axes as columns, in the order A, B, C).
    """
    com = first / mass
    Sc = second - mass * np.outer(com, com)
    lam, axes = np.linalg.eigh(np.trace(Sc) * np.eye(3) - Sc)
    a, b, c = lam / (mass * R0**2)
    J2 = c - 0.5 * (a + b)
    return dict(A=a, B=b, C=c, J2=J2, C22=0.25 * (b - a), H=J2 / c, I=(a + b + c) / 3.0, com=com,
                axes=axes)


class TwoLayer:
    """
    Two-layer body: core of density rho2 inside a mantle of density rho1.

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
    Range of H on the grid of A, B, C at frac x homogeneous (n = 8: 5 % steps, as Ermakov 2026)
    with C the largest.

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
# Precession (Ermakov 2026, Eq. 1)
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
    Orbit-averaged solar precession of the spin pole (Ermakov 2026, Eq. 1).

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
    As Ermakov (2026, Fig. 2): H (black), J2 (vermillion) and rho1 (grey dashed) over core
    radius and density for the spherical (A) and rescaled-shape (B) cores; rho1 < RHO1_MIN
    shaded. Contour levels are shared by the two panels.
    """
    def levels(key, n):
        vals = np.concatenate([np.where(m["rho1"] >= RHO1_MIN, m[key], np.nan).ravel() for m in maps])
        return MaxNLocator(n).tick_values(np.nanmin(vals), np.nanmax(vals))[1:-1]

    lev_H, lev_J2 = levels("H", 7), levels("J2", 6)
    lev_rho1 = MaxNLocator(4).tick_values(RHO1_MIN, RHO_MEAN)[1:-1]
    fig, axs = plt.subplots(1, 2, figsize=(MCMCPlots.TXT_W, 2.9), sharey=True, layout="constrained")
    X, Y = np.meshgrid(r2s, rho2s / 1e3, indexing="ij")
    for ax, m, title, ext in zip(axs, maps, ("(A) Spherical Core", "(B) Rescaled-Shape Core"), extremes):
        rho1 = np.nan_to_num(m["rho1"], nan=-1e5)  # nan: core fills the body
        ax.contourf(X, Y, rho1, levels=[rho1.min() - 1.0, RHO1_MIN], colors=[MCMCPlots.C_SPAN], zorder=0)
        keep = m["rho1"] >= 0.9 * RHO1_MIN  # H, J2 only where the mantle is admissible (with margin)
        H, J2 = np.where(keep, m["H"], np.nan), np.where(keep, m["J2"], np.nan)
        if np.nanmax(H) - np.nanmin(H) > 1e-3:
            cs = ax.contour(X, Y, H, levels=lev_H, colors="k", linewidths=0.7)
            ax.clabel(cs, fontsize=6, fmt="%.2f", inline_spacing=2)
        else:  # rescaled-shape core: H does not depend on the core (Ermakov 2026, Sec. 4.1)
            ax.text(0.04, 0.05, rf"$H = {np.nanmean(H):.4f}$ everywhere", transform=ax.transAxes,
                    fontsize=7, bbox=dict(fc="white", ec="none", pad=1.5))
        cj = ax.contour(X, Y, J2, levels=lev_J2, colors=MCMCPlots.C_BATCH,
                        linewidths=0.7)
        ax.clabel(cj, fontsize=6, fmt="%.3f", inline_spacing=2)
        cr = ax.contour(X, Y, m["rho1"], levels=lev_rho1, colors="0.55", linewidths=0.6,
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
    for the truth C_bar and the 75-110 % range of C_bar, and the sigma_I = 0.01 requirement.
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
    ax.text(0.98, 0.01, r"$\sigma_{\bar{I}} = 0.01$", transform=ax.get_yaxis_transform(), ha="right",
            va="bottom", fontsize=6.5, color=MCMCPlots.C_TRUTH)
    ax.set_xlabel(r"$\sigma_H / H$ [$\%$]")
    ax.set_ylabel(r"$\sigma_{\bar{I}}$ [-]")
    ax.set_xlim(0, 100 * x[-1])
    ax.set_ylim(0, 1.15 * max(np.max(sig_I), sig_I_noH))
    fig.legend(loc="outside upper center", ncol=2, frameon=False, fontsize=6.5)
    return MCMCPlots.save_figure(fig, name)


def plot_mean_moment(curves, I_true, I_hom, I_min, name="fig_mean_moment"):
    """
    Marginals of I_bar: prior (a), MCMC (a) without and with H, MCMC (b), against the truth,
    the homogeneous body and the minimum of endmember (A).

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
    ax.axvline(I_min, color="0.45", ls="-.", lw=0.9, label=r"Minimum, Endmember (A)")
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




def plot_core_sections(V, F, model, post, truth, name="fig_core_sections", n_draw=80):
    """
    Outer shape and core boundaries in the x-z (meridional) and x-y (equatorial) planes of
    the body frame: truth (dashed), posterior draws of MCMC (b) (blue), the r_min sphere.
    """
    t = np.linspace(0.0, 2.0 * np.pi, 721)
    planes = [("x", "z", np.column_stack([np.cos(t), np.zeros_like(t), np.sin(t)]), (0, 2)),
              ("x", "y", np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)]), (0, 1))]
    fig, axs = plt.subplots(1, 2, figsize=(MCMCPlots.TXT_W, 2.9), layout="constrained")
    r_min = model.g.R.min()
    for ax, (lx, ly, u, (i, j)) in zip(axs, planes):
        R = mesh_radius(V, F, u)
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
        Line2D([], [], color="0.25", lw=0.9, label=f"Surface ({SHAPE_OBJ.name})".replace("_", r"\_")),
        Line2D([], [], color=MCMCPlots.C_TRUTH, ls="--", lw=1.0, label="Core (Truth)"),
        Line2D([], [], color=MCMCPlots.C_POST, lw=0.8, label="Core (Posterior Draws)"),
        Line2D([], [], color="0.45", ls=":", lw=0.7, label=r"$r_{\min}$ Sphere"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False)
    return MCMCPlots.save_figure(fig, name)


# ============================================================
# Summary
# ============================================================


def _row(name, ref, ours, fmt="{:.4f}"):
    r = ref if isinstance(ref, str) else fmt.format(ref)
    o = ours if isinstance(ours, str) else fmt.format(ours)
    return f"  {name:<46s} {r:>18s} {o:>22s}"


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
    Vm, Fm = load_obj(SHAPE_OBJ)
    vol0, first0, _ = polyhedron_integrals(Vm, Fm)
    shift = first0 / vol0  # centroid of the mesh = centre of mass of the homogeneous body
    Vm = Vm - shift
    t0 = time.time()
    grid = AngularGrid(Vm, Fm, N_MU, N_PHI)
    t_grid = time.time() - t0
    V = grid.outer[0]
    r_min = grid.R.min() / 1e3
    hom = inertia_quantities(*(RHO_MEAN * np.asarray(x) for x in grid.outer))
    exact = inertia_quantities(*(RHO_MEAN * np.asarray(x) for x in polyhedron_integrals(Vm, Fm)))
    spin_C = np.degrees(np.arccos(abs(hom["axes"][2, 2])))  # body z (spin) vs axis of C
    print(f"[Shape] {SHAPE_OBJ.name}: {len(Vm)} vertices, {len(Fm)} faces, centroid shift "
          f"{np.linalg.norm(shift) / 1e3:.3f} km; {grid.w.size} nodes ({t_grid:.2f} s), V = {V:.5e} m^3 "
          f"(exact {vol0:.5e}), R_v = {grid.r_v / 1e3:.3f} km, r_min = {r_min:.2f} km, "
          f"r_max = {grid.R.max() / 1e3:.2f} km")
    print("[Homogeneous] " + ", ".join(f"{k} = {hom[k]:.4f} (exact {exact[k]:.4f})"
                                       for k in ("A", "B", "C", "J2", "C22", "H", "I"))
          + f"; spin axis vs C axis {spin_C:.2f} deg")
    model = TwoLayer(grid, RHO_MEAN)
    t0 = time.time()
    for _ in range(200):
        model(*TRUTH)
    print(f"[Model] two-layer moments: {(time.time() - t0) / 200 * 1e3:.3f} ms per call")

    # --- 2. endmember maps ---
    r2s, rho2s = np.linspace(*GRID_R2[:2], GRID_R2[2]), np.linspace(*GRID_RHO2[:2], GRID_RHO2[2])
    maps = [endmember_map(model, w, r2s, rho2s) for w in (0.0, 1.0)]
    ok_A = maps[0]["rho1"] >= RHO1_MIN
    minH_A, kH_A = masked_extreme(maps[0]["H"], ok_A, np.nanargmin)
    maxH_A, _ = masked_extreme(maps[0]["H"], ok_A, np.nanargmax)
    minI_A, kI_A = masked_extreme(maps[0]["I"], ok_A, np.nanargmin)
    minI_Ar, _ = masked_extreme(maps[0]["I"], ok_A & (rho2s[None, :] <= RHO2_ROCK), np.nanargmin)
    ok_B = maps[1]["rho1"] >= RHO1_MIN
    minI_B, kI_B = masked_extreme(maps[1]["I"], ok_B, np.nanargmin)
    dH_B = np.nanmax(np.abs(np.where(ok_B, maps[1]["H"], np.nan) - hom["H"]))
    minJ2_B, _ = masked_extreme(maps[1]["J2"], ok_B, np.nanargmin)
    H_grid = moment_grid_H_range(hom)
    print(f"[Endmember A] rho1 >= {RHO1_MIN:.0f}: H in [{minH_A:.4f}, {maxH_A:.4f}], "
          f"min I = {minI_A:.4f} ({minI_Ar:.4f} with rho2 <= {RHO2_ROCK:.0f})")
    print(f"[Endmember B] max |H - H_hom| = {dH_B:.1e}, min J2 = {minJ2_B:.4f}, min I = {minI_B:.4f}")
    print(f"[Grid 75-110 %] H in [{H_grid[0]:.3f}, {H_grid[1]:.3f}]")

    # --- 5. precession of the homogeneous body ---
    prec = precession(hom["H"])
    print(f"[Precession] obliquity {prec['eps']:.2f} deg, period {prec['period']:.3f} My, "
          f"d alpha/dt = {prec['dra']:.1f}, d delta/dt = {prec['ddec']:.1f}, "
          f"rate {prec['rate']:.1f} mas/yr")

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
            shape=SHAPE_OBJ.name, n_mu=N_MU, n_phi=N_PHI, rho_mean=RHO_MEAN, rho1_min=RHO1_MIN,
        )
        MCMCPlots.write_output(fig_dir / "chain.npz", buf.getvalue(), tag="Chain")

    # --- posteriors (a) ---
    flat_A = [with_mean_moment(flat_posterior(ch.astype(float), tau)[0]) for ch, tau in zip(chains_A, taus_A)]
    sig_I = np.array([np.std(x[:, 3]) for x in flat_A[:-1]])
    sig_I_noH = np.std(flat_A[-1][:, 3])
    k_ref = int(np.argmin(np.abs(np.array(SIGMA_H_SWEEP) - SIGMA_H_REL)))
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
            (flat_A[k_ref][:, 3], r"MCMC (a), $+ H$ (%.1f$\%%$)" % (100 * SIGMA_H_REL), MCMCPlots.C_POST, "-",
             True),
            (D_post[:, 0], "MCMC (b), Two-Layer", MCMCPlots.C_SC, "-", False),
        ],
        d_true["I"], hom["I"], minI_A,
    )
    MCMCPlots.configure(SYM_A, ["", "", "", ""], np.ones(4))
    a_true = np.append(theta_A_true, d_true["I"])
    MCMCPlots.plot_corner(flat_A[k_ref], true_theta=a_true, prior=prior_A, prior_label="Prior",
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
    plot_core_sections(Vm, Fm, model, sub, TRUTH)

    # --- summary ---
    head = f"  {'Quantity':<46s} {'Reference':>18s} {'This script':>22s}"
    lines = [
        f"269 Justitia moments of inertia (analysis of Ermakov 2026, PSJ 7:170): {Path(__file__).name}",
        f"Shape {SHAPE_OBJ.name}: {len(Vm)} vertices, {len(Fm)} faces, recentred by "
        f"{np.linalg.norm(shift) / 1e3:.3f} km; quadrature {N_MU} x {N_PHI}; R0 = {R0 / 1e3:.0f} km; "
        f"mean density {RHO_MEAN:.0f} kg/m^3 (assumed)",
        f"V = {V:.5e} m^3, R_v = {grid.r_v / 1e3:.3f} km, r_min = {r_min:.2f} km, "
        f"r_max = {grid.R.max() / 1e3:.2f} km",
        "",
        "1. Homogeneous body (reference: exact polyhedron integrals)",
        head,
        _row("volume [m^3]", f"{vol0:.5e}", f"{V:.5e}", "{}"),
    ]
    for k in ("A", "B", "C", "J2", "C22", "H", "I"):
        lines.append(_row(k + ("_bar" if k in ("A", "B", "C", "I") else ""), exact[k],
                          f"{hom[k]:.4f} ({100 * (hom[k] / exact[k] - 1):+.3f} %)"))
    lines += [
        _row("spin axis (body z) vs C axis [deg]", "0 (principal-axis rotation)", spin_C, "{:.2f}"),
        "   a relaxed rotator spins about C: the offset measures the inconsistency of the shape (lightcurve",
        "   + occultation model), the IAU pole and uniform density; a rendezvous gravity field and pole fix it",
        "",
        "2. Two-layer endmembers (as Ermakov 2026, Sec. 4.1, Fig. 2)",
        head,
        _row("r_min, core exposure onset [km]", "-", r_min, "{:.2f}"),
        _row(f"(A) H range, rho1 >= {RHO1_MIN:.0f}", "-", f"{minH_A:.4f} - {maxH_A:.4f}"),
        _row(f"(A) min I_bar, rho2 <= {GRID_RHO2[1]:.0f}", "-", minI_A),
        _row(f"(A) min I_bar, rho2 <= {RHO2_ROCK:.0f}", "-", minI_Ar),
        _row("(A) min I_bar at r2 [km], rho2 [kg/m^3]", "-", f"{r2s[kI_A[0]]:.1f}, {rho2s[kI_A[1]]:.0f}"),
        _row("(B) H change vs homogeneous", "0 (exact)", f"max {dH_B:.1e}"),
        _row("(B) min J2", "< homogeneous", f"{minJ2_B:.4f}"),
        _row("(B) min I_bar", "-", minI_B),
        _row("H range, A, B, C at 75-110 % (C largest)", "-", f"{H_grid[0]:.3f} - {H_grid[1]:.3f}"),
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
        f"  {'requirement sigma_I = 0.01':<46s} {'Eq. 13':>18s} {'MCMC (a)':>22s}",
        _row("sigma_H/H, C_bar 75 % of homogeneous", f"{100 * sigH_needed[0]:.2f} %", "-", "{}"),
        _row("sigma_H/H, C_bar homogeneous", f"{100 * sigH_needed[1]:.2f} %", "-", "{}"),
        _row("sigma_H/H, C_bar 110 % of homogeneous", f"{100 * sigH_needed[2]:.2f} %", "-", "{}"),
        _row("sigma_H/H, truth C_bar", f"{100 * sigH_needed[3]:.2f} %", f"{100 * sigH_mcmc:.2f} %", "{}"),
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
    rate_req = sigH_needed[1] * prec["rate"]
    lines += [
        "",
        "5. Precession (Ermakov 2026, Eq. 1), homogeneous body",
        f"   pole RA {POLE_RA:.4f}, Dec {POLE_DEC:.4f} deg, spin {SPIN_RATE} deg/day "
        f"(P = {360.0 / SPIN_RATE * 24:.2f} h); orbit a {ORBIT['a']} AU, e {ORBIT['e']}, "
        f"i {ORBIT['i']} deg, node {ORBIT['node']} deg (osculating, 2034-01-01)",
        _row("obliquity [deg]", "-", prec["eps"], "{:.2f}"),
        _row("precession period [My]", "-", prec["period"], "{:.3f}"),
        _row("d alpha / dt [mas/yr]", "-", prec["dra"], "{:.1f}"),
        _row("d delta / dt [mas/yr]", "-", prec["ddec"], "{:.1f}"),
        _row("pole drift rate [mas/yr]", "-", prec["rate"], "{:.1f}"),
        _row("pole drift rate, MCMC (b) H [mas/yr]", "-",
             f"{np.median(prec_B['rate']):.1f} +/- {np.std(prec_B['rate']):.1f}", "{}"),
        f"   the pole drift rate is proportional to H: sigma_H / H = {100 * SIGMA_H_REL:.1f} % <-> "
        f"{SIGMA_H_REL * prec['rate']:.1f} mas/yr; sigma_I = 0.01 (C_bar homogeneous) needs "
        f"sigma_H / H = {100 * sigH_needed[1]:.2f} % <-> {rate_req:.1f} mas/yr",
        f"   ({rate_req * 10:.0f} mas = {np.radians(rate_req * 10 / 3.6e6) * grid.r_v:.2f} m on the "
        f"surface over a 10-yr baseline)",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    MCMCPlots.write_output(fig_dir / "summary.txt", text.encode(), tag="Summary")
    print(f"[Done] {time.time() - t_start:.0f} s; outputs in {fig_dir}")
