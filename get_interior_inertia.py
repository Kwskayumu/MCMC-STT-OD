r"""
Interior of a two-shape asteroid (polyhedral mantle + ellipsoidal core) from its
spherical-harmonic gravity field and its rotation state, by MCMC (emcee).

Standalone: no orbit propagation, no MCMCModel. Only the figures reuse the house style of
MCMCPlots.py (configure, plot_corner, save_figure, write_output).

Model ("two shapes")
--------------------
* Outer shape: the polyhedron of SHAPE_OBJ (Bennu by default), filled with a mantle of
  uniform density rho_m.
* Inner shape: a triaxial ellipsoid (semi-axes a_c, b_c, c_c along the shape-frame axes,
  centre d = (dx, dy, dz)) of uniform density rho_c, replacing the mantle where it sits.
  It must stay inside the outer surface below a crust of at least CRUST_MIN.

  theta = [rho_m, rho_c, a_c, b_c, c_c, dx, dy, dz]       ([g/cm^3] x 2, [m] x 6)

Every observable is linear in the two densities: any mass integral of the body is
rho_m * (outer shape) + (rho_c - rho_m) * (core), with unit-density integrals computed by
exact quadrature (polynomial integrands):
  - outer shape: origin-apex tetrahedra of the mesh, Duffy (collapsed) Gauss rule,
    computed once;
  - core: Gauss rule on the unit ball mapped to the ellipsoid, every likelihood call.

Observables
-----------
Gravity (field referenced to the shape-model origin, i.e. the centre of figure):
  * GM;
  * fully normalized C_nm, S_nm for n = 1..N_DEG. Degree 1 is the centre-of-mass offset
    (C_10 = z_cm / (sqrt(3) R), ...), with sigma from SIGMA_COM; degree >= 2 has SIGMA_CS.
  Gravity gives M, the centre of mass and the trace-free part of the inertia tensor
  (C_20, C_22, C_21, S_21, S_22), but never its trace: C / (M R^2) is not observable from
  gravity alone.
Rotation (inertia tensor about the centre of mass, principal moments A <= B <= C):
  * USE_POLE: the spin pole of a principal-axis rotator is the axis of maximum inertia; the
    observed pole direction (shape frame) constrains e_C (2 angle residuals).
  * USE_TUMBLING: a non-principal-axis rotator (cf. Apophis, Toutatis) gives the dynamic
    moment ratios A/C, B/C from the light-curve / radar / spacecraft inversion, plus the
    long axis e_A in the shape frame (2 angle residuals).
  A/C and B/C together with C_20 and C_22 fix the trace, hence the polar moment factor
  C / (M R_v^2): this is what rotation adds to gravity.

The synthetic data are the truth model (TRUTH) plus Gaussian noise (seed SEED_DATA).

Outputs (results/<script name>_<YYMMDDHHMMSS>/)
-----------------------------------------------
chain.npz (full chain, log-probabilities, data, settings; replot with LOAD_CHAIN),
summary.txt (derived inertia quantities: truth, homogeneous body, prior, posterior), and
one PDF per figure: traces, parameter corner (+ prior), inertia corner (+ prior), inertia
marginals vs homogeneous body, core cross-sections, normalized residuals.
"""

from __future__ import annotations

import io
import math
import time
from datetime import datetime
from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.optimize import least_squares

import MCMCPlots

MCMCPlots.SHOW_FIGS = True  # plt.show() after each figure is saved

HERE = Path(__file__).resolve().parent

# ============================================================
# Settings
# ============================================================

# --- Outer shape ---
SHAPE_OBJ = HERE / "data" / "shape_models" / "Bennu2019.obj"  # body-fixed shape model
SHAPE_UNIT = 1.0e3  # OBJ length unit -> m (the OBJ is in km)
BODY = "Bennu"
R_REF = 290.0  # SH reference radius [m]
N_DEG = 4  # maximum SH degree of the gravity data
G_CONST = 6.67430e-11  # [m^3 / (kg s^2)]

# --- Truth (synthetic data): underdense, slightly off-centre core ---
#            rho_m  rho_c   a_c    b_c    c_c    dx    dy    dz
TRUTH = np.array([1.26, 0.80, 140.0, 125.0, 100.0, 12.0, -8.0, 6.0])

# --- Measurement noise ---
SEED_DATA = 2026  # noise realization of the synthetic data
SIGMA_GM = 6.0e-3  # GM [m^3/s^2]
SIGMA_COM = 1.0  # centre-of-mass offset per axis [m] -> degree-1 coefficients
SIGMA_CS = {2: 2.0e-5, 3: 5.0e-5, 4: 1.0e-4}  # fully normalized C_nm, S_nm, by degree
USE_POLE = True  # spin pole = axis of maximum inertia
SIGMA_POLE_DEG = 0.2  # pole direction in the shape frame [deg]
USE_TUMBLING = True  # non-principal-axis rotation: A/C, B/C and long axis
SIGMA_RATIO = 2.0e-3  # A/C and B/C [-]
SIGMA_LONG_AXIS_DEG = 0.5  # long (minimum-inertia) axis in the shape frame [deg]

# --- Prior: uniform box + core inside the outer shape ---
PRIOR_LO = np.array([0.5, 0.2, 30.0, 30.0, 30.0, -80.0, -80.0, -80.0])
PRIOR_HI = np.array([3.0, 4.0, 230.0, 230.0, 230.0, 80.0, 80.0, 80.0])
CRUST_MIN = 10.0  # minimum mantle thickness above the core [m]

# --- MCMC ---
N_WALKERS = 48
N_STEPS = 20_000
BURN_IN = None  # None: 5 x max autocorrelation time
THIN = None  # None: max(1, min autocorrelation time / 2)
N_PROC = 1  # > 1: multiprocessing pool (the likelihood is ~0.1 ms, so 1 is usually best)
N_STARTS = 12  # multi-start least squares from prior draws -> walker initialization
LOAD_CHAIN = None  # e.g. "results/get_interior_inertia_<stamp>/chain.npz" (replot only)

N_DERIVED = 4000  # posterior samples mapped to derived quantities
N_PRIOR = 20_000  # prior samples for the prior overlays

PAR_NAMES = ["rho_m", "rho_c", "a_c", "b_c", "c_c", "dx", "dy", "dz"]
PARAM_SYM = [
    r"\rho_{\mathrm{m}}",
    r"\rho_{\mathrm{c}}",
    r"a_{\mathrm{c}}",
    r"b_{\mathrm{c}}",
    r"c_{\mathrm{c}}",
    r"\Delta x_{\mathrm{c}}",
    r"\Delta y_{\mathrm{c}}",
    r"\Delta z_{\mathrm{c}}",
]
PARAM_UNIT = ["g/cm$^3$", "g/cm$^3$", "m", "m", "m", "m", "m", "m"]

DERIVED_NAMES = ["C/(M Rv^2)", "A/C", "B/C", "x_cm", "y_cm", "z_cm", "f_core"]
DERIVED_SYM = [
    r"C/(M R_{\mathrm{v}}^2)",
    r"A/C",
    r"B/C",
    r"x_{\mathrm{CM}}",
    r"y_{\mathrm{CM}}",
    r"z_{\mathrm{CM}}",
    r"f_{\mathrm{c}}",
]
DERIVED_UNIT = ["", "", "", "m", "m", "m", ""]


# ============================================================
# Shape model
# ============================================================


def load_shape(path, unit=1.0):
    """
    Read a Wavefront OBJ as a closed triangle mesh with outward normals.

    Parameters
    ----------
    path : str or Path
        OBJ file ('v' and 'f' records; polygons are fan-triangulated).
    unit : float
        Factor from the OBJ length unit to metres.

    Returns
    -------
    V : ndarray, shape (n_v, 3)
        Vertices [m].
    T : ndarray of int, shape (n_t, 3)
        Triangles, oriented so that the signed volume is positive.
    """
    verts, tris = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(p) for p in line.split()[1:4]])
            elif line.startswith("f "):
                idx = []
                for p in line.split()[1:]:
                    k = int(p.split("/")[0])
                    idx.append(k - 1 if k > 0 else len(verts) + k)
                tris += [[idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1)]
    V = np.asarray(verts, dtype=float) * unit
    T = np.asarray(tris, dtype=int)
    if np.einsum("ij,ij->i", V[T[:, 0]], np.cross(V[T[:, 1]], V[T[:, 2]])).sum() < 0.0:
        T = T[:, [0, 2, 1]]
    return V, T


class RadialLimit:
    """
    Inner radius of the outer surface as a function of direction from the shape origin.

    Directions are binned on an equal-area grid (uniform in sin(latitude) and longitude);
    each bin holds the smallest radius of the vertices and face centroids that fall in it,
    so the limit is conservative where the surface is rough or overhangs. Empty bins take
    the minimum of their neighbours.

    Parameters
    ----------
    V : ndarray, shape (n_v, 3)
        Vertices [m].
    T : ndarray of int, shape (n_t, 3)
        Triangles.
    n_z, n_lon : int
        Bins in sin(latitude) and longitude.
    """

    def __init__(self, V, T, n_z=90, n_lon=180):
        self.n_z, self.n_lon = int(n_z), int(n_lon)
        P = np.vstack([V, V[T].mean(axis=1)])
        iz, il, r = self._bins(P)
        R = np.full((self.n_z, self.n_lon), np.inf)
        np.minimum.at(R, (iz, il), r)
        while np.isinf(R).any():  # fill empty bins from their neighbours
            nb = np.stack(
                [np.roll(R, s, axis=ax) for ax in (0, 1) for s in (1, -1)], axis=0
            ).min(axis=0)
            R = np.where(np.isinf(R), nb, R)
        self.R = R

    def _bins(self, P):
        r = np.linalg.norm(P, axis=-1)
        sz = np.divide(P[..., 2], r, out=np.zeros_like(r), where=r > 0)
        lon = np.arctan2(P[..., 1], P[..., 0])
        iz = np.clip(((sz + 1.0) * 0.5 * self.n_z).astype(int), 0, self.n_z - 1)
        il = np.clip(((lon + np.pi) / (2 * np.pi) * self.n_lon).astype(int), 0, self.n_lon - 1)
        return iz, il, r

    def margin(self, P):
        """
        Distance below the surface of each point, along its direction from the origin.

        Parameters
        ----------
        P : ndarray, shape (..., 3)
            Points [m].

        Returns
        -------
        ndarray, shape (...)
            Surface radius minus point radius [m] (negative outside).
        """
        iz, il, r = self._bins(P)
        return self.R[iz, il] - r


def fibonacci_sphere(n):
    """n quasi-uniform unit vectors (Fibonacci lattice), shape (n, 3)."""
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0 * k / n
    phi = np.pi * (1.0 + 5.0**0.5) * k
    s = np.sqrt(1.0 - z * z)
    return np.column_stack([s * np.cos(phi), s * np.sin(phi), z])


# ============================================================
# Mass integrals: solid harmonics, polyhedron, ellipsoid
# ============================================================


def solid_harmonics(X, n_max, r_ref):
    r"""
    Fully normalized solid harmonics at points X.

    V_nm = (r/R)^n Pbar_nm(sin phi) [cos m lambda, sin m lambda], evaluated with Cartesian
    recursions (polynomials in x/R, y/R, z/R; no division by r, so r = 0 is fine).
    Pbar_nm carries sqrt((2 - delta_m0)(2n + 1)(n - m)!/(n + m)!), no Condon-Shortley phase,
    so that for a body of mass M

        Cbar_nm + i Sbar_nm = 1 / ((2n + 1) M) * int V_nm dm.

    Parameters
    ----------
    X : ndarray, shape (Q, 3)
        Points [m].
    n_max : int
        Maximum degree.
    r_ref : float
        Reference radius [m].

    Returns
    -------
    Vc, Vs : ndarray, shape (n_max + 1, n_max + 1, Q)
        Cosine and sine parts (lower triangle n >= m filled, the rest zero).
    """
    x, y, z = (np.asarray(X, dtype=float) / r_ref).T
    r2 = x * x + y * y + z * z
    Vc = np.zeros((n_max + 1, n_max + 1, x.size))
    Vs = np.zeros_like(Vc)
    Vc[0, 0] = 1.0
    for m in range(1, n_max + 1):  # sectoral
        f = math.sqrt(3.0) if m == 1 else math.sqrt((2 * m + 1) / (2 * m))
        Vc[m, m] = f * (x * Vc[m - 1, m - 1] - y * Vs[m - 1, m - 1])
        Vs[m, m] = f * (x * Vs[m - 1, m - 1] + y * Vc[m - 1, m - 1])
    for m in range(0, n_max):  # first subdiagonal
        f = math.sqrt(2 * m + 3)
        Vc[m + 1, m] = f * z * Vc[m, m]
        Vs[m + 1, m] = f * z * Vs[m, m]
    for m in range(0, n_max + 1):  # vertical recursion
        for n in range(m + 2, n_max + 1):
            a = math.sqrt((2 * n + 1) * (2 * n - 1) / ((n - m) * (n + m)))
            b = math.sqrt(
                (2 * n + 1) * (n + m - 1) * (n - m - 1) / ((2 * n - 3) * (n - m) * (n + m))
            )
            Vc[n, m] = a * z * Vc[n - 1, m] - b * r2 * Vc[n - 2, m]
            Vs[n, m] = a * z * Vs[n - 1, m] - b * r2 * Vs[n - 2, m]
    return Vc, Vs


def _gauss01(n):
    """n-point Gauss-Legendre nodes and weights on [0, 1]."""
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


class Moments:
    """
    Mass integrals of a body, linear in density.

    Attributes
    ----------
    mass : float
        int dm (a volume [m^3] for unit density).
    first : ndarray, shape (3,)
        int x dm.
    second : ndarray, shape (3, 3)
        int x x^T dm.
    Ic, Is : ndarray, shape (n_max + 1, n_max + 1)
        int V_nm dm (solid_harmonics), so that Cbar_nm = Ic[n, m] / ((2n + 1) mass).
    """

    def __init__(self, mass, first, second, Ic, Is):
        self.mass, self.first, self.second, self.Ic, self.Is = mass, first, second, Ic, Is

    def combine(self, k_self, other, k_other):
        """k_self * self + k_other * other (e.g. densities of mantle and core)."""
        return Moments(
            *(
                k_self * getattr(self, a) + k_other * getattr(other, a)
                for a in ("mass", "first", "second", "Ic", "Is")
            )
        )


def polyhedron_moments(V, T, n_max, r_ref, chunk=4000):
    """
    Unit-density mass integrals of a closed polyhedron, exact up to degree n_max.

    The body is split into the signed tetrahedra (origin, v0, v1, v2) of its faces. On each
    one, x = t [(1 - s) v0 + s (1 - q) v1 + s q v2] maps the unit cube onto the tetrahedron
    with dV = 6 vol t^2 s dt ds dq (Duffy), and a tensor Gauss rule integrates the
    polynomial integrands (degree <= n_max) exactly.

    Parameters
    ----------
    V : ndarray, shape (n_v, 3)
        Vertices [m].
    T : ndarray of int, shape (n_t, 3)
        Outward-oriented triangles.
    n_max : int
        Maximum SH degree (>= 2, for the inertia tensor).
    r_ref : float
        Reference radius [m].
    chunk : int
        Faces per vectorized batch.

    Returns
    -------
    Moments
    """
    deg = max(n_max, 2)
    t, wt = _gauss01((deg + 4) // 2)  # degree deg + 2 in t
    s, ws = _gauss01((deg + 3) // 2)  # degree deg + 1 in s
    q, wq = _gauss01((deg + 2) // 2)  # degree deg in q
    TT, SS, QQ = (g.ravel() for g in np.meshgrid(t, s, q, indexing="ij"))
    W = (np.einsum("i,j,k->ijk", wt * t**2, ws * s, wq)).ravel() * 6.0
    c0, c1, c2 = TT * (1.0 - SS), TT * SS * (1.0 - QQ), TT * SS * QQ  # barycentric

    mass, first, second = 0.0, np.zeros(3), np.zeros((3, 3))
    Ic = np.zeros((n_max + 1, n_max + 1))
    Is = np.zeros_like(Ic)
    for i in range(0, len(T), chunk):
        v0, v1, v2 = (V[T[i : i + chunk, k]] for k in range(3))
        vol = np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0
        P = (
            c0[None, :, None] * v0[:, None, :]
            + c1[None, :, None] * v1[:, None, :]
            + c2[None, :, None] * v2[:, None, :]
        ).reshape(-1, 3)
        w = (vol[:, None] * W[None, :]).ravel()
        mass += w.sum()
        first += w @ P
        second += (P * w[:, None]).T @ P
        Vc, Vs = solid_harmonics(P, n_max, r_ref)
        Ic += Vc @ w
        Is += Vs @ w
    return Moments(mass, first, second, Ic, Is)


class BallRule:
    """
    Gauss rule on the unit ball, exact for polynomials of degree <= n_max.

    Gauss-Legendre in r (weight r^2 included) and in cos(theta), trapezoid in longitude.

    Attributes
    ----------
    U : ndarray, shape (Q, 3)
        Nodes.
    w : ndarray, shape (Q,)
        Weights (sum 4 pi / 3).
    """

    def __init__(self, n_max):
        r, wr = _gauss01(n_max // 2 + 3)
        mu, wmu = np.polynomial.legendre.leggauss(n_max // 2 + 2)
        n_phi = n_max + 2
        phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
        R, MU, PHI = (g.ravel() for g in np.meshgrid(r, mu, phi, indexing="ij"))
        sth = np.sqrt(1.0 - MU**2)
        self.U = np.column_stack([R * sth * np.cos(PHI), R * sth * np.sin(PHI), R * MU])
        self.w = np.einsum("i,j,k->ijk", wr * r**2, wmu, np.full(n_phi, 2 * np.pi / n_phi)).ravel()


def ellipsoid_moments(axes, center, n_max, r_ref, rule):
    """
    Unit-density mass integrals of an ellipsoid aligned with the frame axes.

    Mass, first and second moments are analytic; the solid-harmonic integrals use `rule`
    mapped to the ellipsoid (x = center + axes * u), which is exact for degree <= n_max.

    Parameters
    ----------
    axes : array_like, shape (3,)
        Semi-axes along x, y, z [m].
    center : array_like, shape (3,)
        Centre [m].
    n_max : int
        Maximum SH degree.
    r_ref : float
        Reference radius [m].
    rule : BallRule

    Returns
    -------
    Moments
    """
    axes, center = np.asarray(axes, float), np.asarray(center, float)
    vol = 4.0 / 3.0 * np.pi * np.prod(axes)
    Vc, Vs = solid_harmonics(center + rule.U * axes, n_max, r_ref)
    w = rule.w * np.prod(axes)
    return Moments(
        vol,
        vol * center,
        vol * (np.diag(axes**2) / 5.0 + np.outer(center, center)),
        Vc @ w,
        Vs @ w,
    )


def inertia_tensor(mom):
    """
    Inertia tensor about the centre of mass and its principal axes.

    Parameters
    ----------
    mom : Moments
        Mass integrals [kg, kg m, kg m^2].

    Returns
    -------
    com : ndarray, shape (3,)
        Centre of mass [m].
    I : ndarray, shape (3, 3)
        Inertia tensor about the centre of mass [kg m^2].
    moments : ndarray, shape (3,)
        Principal moments A <= B <= C [kg m^2].
    axes : ndarray, shape (3, 3)
        Principal axes as columns (e_A, e_B, e_C).
    """
    com = mom.first / mom.mass
    Sc = mom.second - mom.mass * np.outer(com, com)
    I = np.trace(Sc) * np.eye(3) - Sc
    moments, axes = np.linalg.eigh(I)
    return com, I, moments, axes


def cs_index(n_max):
    """(kind, n, m) of the observed coefficients: degree 1..n_max, C_nm for m >= 0, S_nm for m >= 1."""
    out = []
    for n in range(1, n_max + 1):
        for m in range(n + 1):
            out.append(("C", n, m))
            if m > 0:
                out.append(("S", n, m))
    return out


def _tangent_basis(u):
    """Two unit vectors orthogonal to the unit vector u (and to each other)."""
    a = np.eye(3)[np.argmin(np.abs(u))]
    t1 = np.cross(u, a)
    t1 /= np.linalg.norm(t1)
    return t1, np.cross(u, t1)


def _axis_offset(e, u_obs, t1, t2):
    """Small-angle offset [rad] of the axis e (sign-free) from u_obs, along t1 and t2."""
    e = e if e @ u_obs >= 0.0 else -e
    return np.array([e @ t1, e @ t2])


# ============================================================
# Forward model and posterior
# ============================================================


class InteriorModel:
    """
    Forward model, observations and log posterior of the mantle + core body.

    Picklable (no closures), so it can be handed to a multiprocessing pool.

    Parameters
    ----------
    outer : Moments
        Unit-density integrals of the outer shape.
    radial : RadialLimit
        Inner radius of the outer surface (core containment).
    n_max, r_ref : int, float
        SH degree and reference radius [m].
    """

    def __init__(self, outer, radial, n_max, r_ref):
        self.outer, self.radial = outer, radial
        self.n_max, self.r_ref = n_max, r_ref
        self.rule = BallRule(n_max)
        self.sphere = fibonacci_sphere(600)
        self.idx = cs_index(n_max)
        self.obs = None

    # --- forward model ---
    def moments(self, theta):
        """Mass integrals [SI] of the body for theta."""
        rho_m, rho_c = 1e3 * theta[0], 1e3 * theta[1]
        core = ellipsoid_moments(theta[2:5], theta[5:8], self.n_max, self.r_ref, self.rule)
        return self.outer.combine(rho_m, core, rho_c - rho_m)

    def predict(self, theta):
        """
        Predicted observables for theta.

        Returns
        -------
        dict
            GM [m^3/s^2], cs (observed coefficient vector, cs_index order), com [m],
            I [kg m^2], moments (A, B, C) [kg m^2], axes (columns e_A, e_B, e_C), mass [kg].
        """
        mom = self.moments(theta)
        cs = np.array(
            [
                (mom.Ic if k == "C" else mom.Is)[n, m] / ((2 * n + 1) * mom.mass)
                for k, n, m in self.idx
            ]
        )
        com, I, moments, axes = inertia_tensor(mom)
        return dict(
            GM=G_CONST * mom.mass, cs=cs, com=com, I=I, moments=moments, axes=axes, mass=mom.mass
        )

    def core_inside(self, theta):
        """True if the core lies below the outer surface by at least CRUST_MIN."""
        P = theta[5:8] + self.sphere * theta[2:5]
        return bool(np.all(self.radial.margin(P) >= CRUST_MIN))

    # --- data ---
    def observation_vector(self, pred):
        """Raw observables (no rotation-axis angles) of a prediction, for simulation."""
        return dict(GM=pred["GM"], cs=pred["cs"].copy())

    def set_observations(self, obs):
        """
        Install the observations.

        Parameters
        ----------
        obs : dict
            GM, sig_GM, cs, sig_cs, and the rotation data used by the USE_* switches:
            pole (unit vector), sig_pole [rad], ratios (A/C, B/C), sig_ratio,
            long_axis (unit vector), sig_long [rad].
        """
        self.obs = dict(obs)
        self.obs["pole_t"] = _tangent_basis(obs["pole"])
        self.obs["long_t"] = _tangent_basis(obs["long_axis"])

    def residual_names(self):
        """Labels of the entries of residuals() (LaTeX, for the residual figure)."""
        names = [r"$GM$"] + [rf"${k}_{{{n}{m}}}$" for k, n, m in self.idx]
        if USE_POLE:
            names += [r"pole$_1$", r"pole$_2$"]
        if USE_TUMBLING:
            names += [r"$A/C$", r"$B/C$", r"$\hat e_{A,1}$", r"$\hat e_{A,2}$"]
        return names

    def residuals(self, theta, pred=None):
        """Normalized residuals (observed - model) / sigma of every observable."""
        o = self.obs
        p = self.predict(theta) if pred is None else pred
        r = [[(o["GM"] - p["GM"]) / o["sig_GM"]], (o["cs"] - p["cs"]) / o["sig_cs"]]
        if USE_POLE:
            r.append(-_axis_offset(p["axes"][:, 2], o["pole"], *o["pole_t"]) / o["sig_pole"])
        if USE_TUMBLING:
            A, B, C = p["moments"]
            r.append((o["ratios"] - np.array([A / C, B / C])) / o["sig_ratio"])
            r.append(-_axis_offset(p["axes"][:, 0], o["long_axis"], *o["long_t"]) / o["sig_long"])
        return np.concatenate(r)

    # --- posterior ---
    def log_prior(self, theta):
        """Uniform box (PRIOR_LO, PRIOR_HI) and core containment: 0 or -inf."""
        if np.any(theta < PRIOR_LO) or np.any(theta > PRIOR_HI):
            return -np.inf
        return 0.0 if self.core_inside(theta) else -np.inf

    def __call__(self, theta):
        """Log posterior (up to a constant)."""
        theta = np.asarray(theta, dtype=float)
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        r = self.residuals(theta)
        return lp - 0.5 * float(r @ r)

    # --- derived quantities ---
    def derived(self, theta):
        """
        Inertia quantities of theta (DERIVED_NAMES order).

        C / (M R_v^2) with R_v the volume-equivalent radius of the outer shape, the moment
        ratios, the centre-of-mass offset from the shape origin [m] and the core mass
        fraction.
        """
        p = self.predict(theta)
        A, B, C = p["moments"]
        r_v = (3.0 * self.outer.mass / (4.0 * np.pi)) ** (1.0 / 3.0)
        m_core = 1e3 * theta[1] * 4.0 / 3.0 * np.pi * np.prod(theta[2:5])
        return np.array(
            [C / (p["mass"] * r_v**2), A / C, B / C, *p["com"], m_core / p["mass"]]
        )


def simulate_observations(model, theta_true, rng):
    """
    Synthetic data: the truth model plus Gaussian noise.

    Rotation axes are perturbed by a random small rotation of the configured angular sigma.

    Returns
    -------
    dict
        Observations for InteriorModel.set_observations.
    """
    p = model.predict(theta_true)
    sig_cs = np.array(
        [
            SIGMA_COM / (math.sqrt(3.0) * R_REF) if n == 1 else SIGMA_CS[n]
            for _, n, _ in model.idx
        ]
    )

    def perturb(u, sig):
        t1, t2 = _tangent_basis(u)
        v = u + sig * (rng.standard_normal() * t1 + rng.standard_normal() * t2)
        return v / np.linalg.norm(v)

    A, B, C = p["moments"]
    e_A, e_C = p["axes"][:, 0], p["axes"][:, 2]
    sig_pole = math.radians(SIGMA_POLE_DEG)
    sig_long = math.radians(SIGMA_LONG_AXIS_DEG)
    return dict(
        GM=p["GM"] + SIGMA_GM * rng.standard_normal(),
        sig_GM=SIGMA_GM,
        cs=p["cs"] + sig_cs * rng.standard_normal(sig_cs.size),
        sig_cs=sig_cs,
        pole=perturb(e_C if e_C[2] >= 0 else -e_C, sig_pole),
        sig_pole=sig_pole,
        ratios=np.array([A / C, B / C]) + SIGMA_RATIO * rng.standard_normal(2),
        sig_ratio=SIGMA_RATIO,
        long_axis=perturb(e_A if e_A[0] >= 0 else -e_A, sig_long),
        sig_long=sig_long,
    )


def sample_prior(model, n, rng, batch=4096):
    """n draws of the prior (uniform box, rejection on core containment)."""
    out = []
    while sum(len(o) for o in out) < n:
        th = PRIOR_LO + (PRIOR_HI - PRIOR_LO) * rng.random((batch, PRIOR_LO.size))
        out.append(th[[model.core_inside(t) for t in th]])
    return np.vstack(out)[:n]


def initialize_walkers(model, n_walkers, n_starts, rng):
    """
    Multi-start bounded least squares from prior draws, then walkers around the best fit.

    Walkers are drawn from N(theta_map, cov) with cov = (J^T J)^-1 at the best fit, and
    redrawn until they have finite posterior.

    Returns
    -------
    p0 : ndarray, shape (n_walkers, ndim)
    theta_map : ndarray, shape (ndim,)
    """
    best, best_lp = None, -np.inf
    width = PRIOR_HI - PRIOR_LO
    for k, x0 in enumerate(sample_prior(model, n_starts, rng)):
        sol = least_squares(
            model.residuals, x0, bounds=(PRIOR_LO, PRIOR_HI), x_scale=width, method="trf",
            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400,
        )
        lp = model(sol.x)
        print(f"[Init] start {k + 1:2d}/{n_starts}: chi2 = {2 * sol.cost:12.3f}  "
              f"log p = {lp:12.3f}")
        if lp > best_lp:
            best, best_lp = sol, lp
    if best is None:
        raise RuntimeError("no start reached a core inside the shape; widen the search")
    try:
        cov = np.linalg.inv(best.jac.T @ best.jac)
        L = np.linalg.cholesky(0.5 * (cov + cov.T))
    except np.linalg.LinAlgError:
        L = np.diag(1e-4 * width)
    p0 = []
    while len(p0) < n_walkers:
        th = best.x + L @ rng.standard_normal(best.x.size)
        if np.isfinite(model(th)):
            p0.append(th)
    print(f"[Init] best fit: chi2 = {2 * best.cost:.3f} for "
          f"{best.fun.size} observables, {best.x.size} parameters")
    return np.array(p0), best.x


# ============================================================
# Figures
# ============================================================


def plot_traces(chain, logp, burn, truth, name="fig_traces"):
    """Walker traces of every parameter and of log p (burn-in shaded, truth dashed)."""
    n_step, n_walk, n_par = chain.shape
    stride = max(1, n_step // 2000)
    it = np.arange(0, n_step, stride)
    ncol = 3
    nrow = int(np.ceil((n_par + 1) / ncol))
    fig, axs = plt.subplots(
        nrow, ncol, figsize=(MCMCPlots.TXT_W, 1.15 * nrow), sharex=True, layout="constrained"
    )
    axs = axs.ravel()
    series = [chain[..., i] for i in range(n_par)] + [logp]
    labels = [MCMCPlots.param_label(i) for i in range(n_par)] + [r"$\log p$"]
    for i, (ax, y) in enumerate(zip(axs, series)):
        ax.axvspan(0, burn, color=MCMCPlots.C_SPAN, lw=0, zorder=0)
        ax.plot(it, y[it], color=MCMCPlots.C_POST, lw=0.3, alpha=0.25, rasterized=True)
        if i < n_par:
            ax.axhline(truth[i], color=MCMCPlots.C_TRUTH, ls="--", lw=0.8)
        else:
            post = y[burn:]
            lo = np.percentile(post, 0.5) - 3.0 * np.std(post)
            ax.set_ylim(lo, post.max() + 0.2 * (post.max() - lo))
        ax.set_ylabel(labels[i])
    for ax in axs[n_par + 1 :]:
        ax.set_visible(False)
    for ax in axs[max(0, n_par + 1 - ncol) : n_par + 1]:
        ax.set_xlabel("Step")
        ax.tick_params(labelbottom=True)
    return MCMCPlots.save_figure(fig, name)


def plot_inertia_marginals(D_post, D_prior, d_true, d_hom, name="fig_inertia_marginals"):
    """Posterior / prior marginals of C/(M R_v^2), A/C, B/C with the truth and the homogeneous body."""
    fig, axs = plt.subplots(1, 3, figsize=(MCMCPlots.TXT_W, 1.9), layout="constrained")
    for k, ax in enumerate(axs):
        x = D_post[:, k]
        lo, hi = np.percentile(x, [0.05, 99.95])
        span = hi - lo
        lo = min(lo, d_true[k], d_hom[k]) - 0.08 * span
        hi = max(hi, d_true[k], d_hom[k]) + 0.08 * span
        bins = np.linspace(lo, hi, 60)
        ax.hist(x, bins=bins, density=True, color=MCMCPlots._tint(MCMCPlots.C_POST, 0.45),
                edgecolor=MCMCPlots.C_POST, lw=0.6, histtype="stepfilled", label="Posterior")
        xp = D_prior[:, k]
        xp = xp[(xp > lo) & (xp < hi)]
        if xp.size > 50:
            h, e = np.histogram(xp, bins=bins, density=False)
            h = h / (len(D_prior) * np.diff(e))
            ax.stairs(h, e, color=MCMCPlots.C_PRIOR, ls="--", lw=0.9, label="Prior")
        ax.axvline(d_true[k], color=MCMCPlots.C_TRUTH, lw=1.0, label="Truth")
        ax.axvline(d_hom[k], color=MCMCPlots.C_BATCH, ls=":", lw=1.2, label="Homogeneous")
        ax.set_xlim(lo, hi)
        ax.set_xlabel(f"${DERIVED_SYM[k]}$")
        ax.set_yticks([])
        ax.grid(False)
    axs[0].set_ylabel("Density")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=4)
    return MCMCPlots.save_figure(fig, name)


def _mesh_section(V, T, axis):
    """Segments (n, 2, 2) of the outer-surface cut by the plane x_axis = 0, in the other two axes."""
    keep = [k for k in range(3) if k != axis]
    d = V[T][:, :, axis]  # (n_t, 3) signed distances
    segs = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        cross = (d[:, i] >= 0.0) != (d[:, j] >= 0.0)  # vertices on the plane count as above
        a, b = V[T[cross, i]], V[T[cross, j]]
        lam = (d[cross, i] / (d[cross, i] - d[cross, j]))[:, None]
        segs.append((np.flatnonzero(cross), a + lam * (b - a)))
    # pair the two crossing points of each cut triangle
    idx = np.concatenate([s[0] for s in segs])
    pts = np.concatenate([s[1] for s in segs])
    order = np.argsort(idx, kind="stable")
    idx, pts = idx[order], pts[order]
    first = np.flatnonzero(np.r_[True, idx[1:] != idx[:-1]])
    ok = (first + 1 < len(idx)) & (idx[np.minimum(first + 1, len(idx) - 1)] == idx[first])
    first = first[ok]
    return np.stack([pts[first][:, keep], pts[first + 1][:, keep]], axis=1)


def plot_core_sections(V, T, post, truth, com_post, com_true, name="fig_core_sections", n_draw=80):
    """Cuts of the outer shape through the origin with the core: truth and posterior draws."""
    from matplotlib.collections import LineCollection

    rng = np.random.default_rng(1)
    draws = post[rng.choice(len(post), size=min(n_draw, len(post)), replace=False)]
    fig, axs = plt.subplots(1, 3, figsize=(MCMCPlots.TXT_W, 2.55), layout="constrained")
    axes_names = ["x", "y", "z"]
    for ax, cut in zip(axs, (2, 1, 0)):  # planes z = 0, y = 0, x = 0
        i, j = [k for k in range(3) if k != cut]
        ax.add_collection(LineCollection(_mesh_section(V, T, cut), colors="0.35", lw=0.6))

        def ellipse(th, **kw):
            s = 1.0 - (th[5 + cut] / th[2 + cut]) ** 2
            if s <= 0.0:
                return
            f = math.sqrt(s)
            ax.add_patch(Ellipse((th[5 + i], th[5 + j]), 2 * th[2 + i] * f, 2 * th[2 + j] * f,
                                 fill=False, **kw))

        for th in draws:
            ellipse(th, ec=MCMCPlots.C_POST, lw=0.4, alpha=0.18)
        ellipse(truth, ec=MCMCPlots.C_TRUTH, lw=1.0, ls="--")
        ax.plot(com_post[:, i], com_post[:, j], ".", ms=0.8, color=MCMCPlots.C_POST,
                alpha=0.3, rasterized=True)
        ax.plot(com_true[i], com_true[j], "*", ms=7, mfc=MCMCPlots.C_TRUTH, mec="white", mew=0.4)
        ax.plot(0, 0, "+", ms=6, color="0.35", mew=0.6)
        ax.set_aspect("equal")
        lim = 1.05 * np.abs(V[:, [i, j]]).max()
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"${axes_names[i]}$ [m]")
        ax.set_ylabel(f"${axes_names[j]}$ [m]")
        ax.set_title(f"${axes_names[cut]} = 0$")
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color="0.35", lw=0.6, label=f"{BODY} Surface"),
        Line2D([], [], color=MCMCPlots.C_TRUTH, ls="--", lw=1.0, label="Core (Truth)"),
        Line2D([], [], color=MCMCPlots.C_POST, lw=0.8, label="Core (Posterior Draws)"),
        Line2D([], [], ls="", marker=".", color=MCMCPlots.C_POST, label="Centre of Mass (Posterior)"),
        Line2D([], [], ls="", marker="*", mfc=MCMCPlots.C_TRUTH, mec="white", ms=7,
               label="Centre of Mass (Truth)"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=5)
    return MCMCPlots.save_figure(fig, name)


def plot_residuals(model, post, theta_hom, name="fig_residuals", n_draw=300):
    """Normalized residuals of every observable: posterior predictive band vs homogeneous body."""
    rng = np.random.default_rng(2)
    draws = post[rng.choice(len(post), size=min(n_draw, len(post)), replace=False)]
    R = np.array([model.residuals(th) for th in draws])
    r_hom = model.residuals(theta_hom)
    names = model.residual_names()
    k = np.arange(len(names))
    lo, med, hi = np.percentile(R, [2.5, 50, 97.5], axis=0)
    fig, ax = plt.subplots(figsize=(MCMCPlots.TXT_W, 2.3), layout="constrained")
    ax.axhspan(-3, 3, color=MCMCPlots.C_SPAN, lw=0, zorder=0)
    ax.vlines(k, lo, hi, color=MCMCPlots.C_POST, lw=1.6, alpha=0.5)
    ax.plot(k, med, "o", ms=2.8, color=MCMCPlots.C_POST, label=r"Posterior (median, 95$\%$)")
    ax.plot(k, r_hom, "x", ms=3.5, mew=0.9, color=MCMCPlots.C_BATCH, label="Homogeneous Body")
    ax.set_yscale("symlog", linthresh=3.0)
    ax.set_xticks(k, names, rotation=90, fontsize=6)
    ax.set_xlim(-0.7, k[-1] + 0.7)
    ax.set_ylabel(r"(Observed $-$ Model) / $\sigma$")
    ax.legend(loc="upper right", ncol=2)
    return MCMCPlots.save_figure(fig, name)


def _fmt(x):
    return f"{x:12.6g}"


def write_summary(path, D_post, D_prior, d_true, d_hom, theta_post, extra_lines):
    """Text table of the parameters and the derived inertia quantities."""
    lines = list(extra_lines) + [""]
    head = f"{'quantity':>14s} {'truth':>12s} {'homogeneous':>12s} {'median':>12s} " \
           f"{'-1sig':>12s} {'+1sig':>12s} {'prior 16%':>12s} {'prior 84%':>12s}"
    lines += ["Derived inertia quantities (posterior median, 16-84 % half-widths)", head]
    for k, nm in enumerate(DERIVED_NAMES):
        q16, q50, q84 = np.percentile(D_post[:, k], [16, 50, 84])
        p16, p84 = np.percentile(D_prior[:, k], [16, 84])
        lines.append(f"{nm:>14s} {_fmt(d_true[k])} {_fmt(d_hom[k])} {_fmt(q50)} "
                     f"{_fmt(q50 - q16)} {_fmt(q84 - q50)} {_fmt(p16)} {_fmt(p84)}")
    lines += ["", "Parameters", f"{'parameter':>14s} {'truth':>12s} {'median':>12s} "
              f"{'-1sig':>12s} {'+1sig':>12s}"]
    for k, nm in enumerate(PAR_NAMES):
        q16, q50, q84 = np.percentile(theta_post[:, k], [16, 50, 84])
        lines.append(f"{nm:>14s} {_fmt(TRUTH[k])} {_fmt(q50)} {_fmt(q50 - q16)} {_fmt(q84 - q50)}")
    text = "\n".join(lines) + "\n"
    print(text)
    MCMCPlots.write_output(path, text.encode(), tag="Summary")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    t_start = time.time()
    MCMCPlots.configure(PARAM_SYM, PARAM_UNIT, np.ones(len(PARAM_SYM)), body=BODY)

    # --- outer shape: unit-density integrals, containment limit ---
    V, T = load_shape(SHAPE_OBJ, SHAPE_UNIT)
    t0 = time.time()
    outer = polyhedron_moments(V, T, N_DEG, R_REF)
    radial = RadialLimit(V, T)
    r_v = (3.0 * outer.mass / (4.0 * np.pi)) ** (1.0 / 3.0)
    print(f"[Shape] {SHAPE_OBJ.name}: {len(V)} vertices, {len(T)} faces, "
          f"volume {outer.mass:.6e} m^3, R_v = {r_v:.2f} m ({time.time() - t0:.1f} s)")

    model = InteriorModel(outer, radial, N_DEG, R_REF)
    if not model.core_inside(TRUTH):
        raise ValueError("TRUTH core is not inside the shape (CRUST_MIN)")

    # --- data ---
    saved = np.load(LOAD_CHAIN, allow_pickle=True) if LOAD_CHAIN else None
    if saved is not None:
        obs = saved["obs"].item()
    else:
        obs = simulate_observations(model, TRUTH, np.random.default_rng(SEED_DATA))
    model.set_observations(obs)
    n_obs = model.residuals(TRUTH).size
    print(f"[Data] {n_obs} observables (GM, {len(model.idx)} SH coefficients to degree {N_DEG}"
          f"{', pole' if USE_POLE else ''}{', A/C, B/C, long axis' if USE_TUMBLING else ''}); "
          f"chi2 at truth = {np.sum(model.residuals(TRUTH) ** 2):.2f}")

    rho_bulk = obs["GM"] / (G_CONST * outer.mass) / 1e3
    theta_hom = np.array([rho_bulk, rho_bulk, 100.0, 100.0, 100.0, 0.0, 0.0, 0.0])
    r_hom = model.residuals(theta_hom)
    print(f"[Data] homogeneous body ({rho_bulk:.4f} g/cm^3): chi2 = {r_hom @ r_hom:.1f}")
    t0 = time.time()
    for _ in range(200):
        model(TRUTH)
    print(f"[Model] log posterior: {(time.time() - t0) / 200 * 1e3:.3f} ms per call")

    fig_dir = MCMCPlots.set_figure_dir(
        HERE / "results" / (Path(__file__).stem + "_" + datetime.now().strftime("%y%m%d%H%M%S"))
    )

    # --- sampling ---
    rng = np.random.default_rng()
    if saved is not None:
        chain, logp = saved["chain"].astype(float), saved["log_prob"].astype(float)
        tau, theta_map = saved["tau"], saved["theta_map"]
        print(f"[MCMC] loaded {LOAD_CHAIN}: chain {chain.shape}")
    else:
        p0, theta_map = initialize_walkers(model, N_WALKERS, N_STARTS, rng)
        moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
        if N_PROC > 1:
            from multiprocessing import Pool

            with Pool(N_PROC) as pool:
                sampler = emcee.EnsembleSampler(N_WALKERS, len(TRUTH), model, moves=moves, pool=pool)
                sampler.run_mcmc(p0, N_STEPS, progress=True)
        else:
            sampler = emcee.EnsembleSampler(N_WALKERS, len(TRUTH), model, moves=moves)
            sampler.run_mcmc(p0, N_STEPS, progress=True)
        chain, logp = sampler.get_chain(), sampler.get_log_prob()
        tau = sampler.get_autocorr_time(tol=0)
        print(f"[MCMC] acceptance {np.mean(sampler.acceptance_fraction):.3f}")
        buf = io.BytesIO()
        np.savez_compressed(
            buf, chain=chain.astype(np.float32), log_prob=logp.astype(np.float32), tau=tau,
            theta_map=theta_map, truth=TRUTH, obs=np.array(obs, dtype=object),
            par_names=np.array(PAR_NAMES), shape=str(SHAPE_OBJ.name), n_deg=N_DEG, r_ref=R_REF,
            use_pole=USE_POLE, use_tumbling=USE_TUMBLING, prior_lo=PRIOR_LO, prior_hi=PRIOR_HI,
            crust_min=CRUST_MIN,
        )
        MCMCPlots.write_output(fig_dir / "chain.npz", buf.getvalue(), tag="Chain")

    burn = int(5 * np.max(tau)) if BURN_IN is None else int(BURN_IN)
    thin = max(1, int(0.5 * np.min(tau))) if THIN is None else int(THIN)
    if burn >= chain.shape[0]:
        print(f"[MCMC] WARNING: burn-in {burn} >= {chain.shape[0]} steps; using half the chain")
        burn = chain.shape[0] // 2
    post = chain[burn::thin].reshape(-1, chain.shape[-1])
    print(f"[MCMC] tau = {np.array2string(tau, precision=1)}; burn-in {burn}, thin {thin}, "
          f"{len(post)} samples ({chain.shape[0] / np.max(tau):.0f} tau)")

    # --- derived inertia quantities ---
    sub = post[np.random.default_rng(3).choice(len(post), min(N_DERIVED, len(post)), replace=False)]
    D_post = np.array([model.derived(th) for th in sub])
    prior = sample_prior(model, N_PRIOR, np.random.default_rng(4))
    D_prior = np.array([model.derived(th) for th in prior])
    d_true, d_hom = model.derived(TRUTH), model.derived(theta_hom)

    # --- figures ---
    plot_traces(chain, logp, burn, TRUTH)
    MCMCPlots.plot_corner(post, true_theta=TRUTH, name="fig_corner")
    MCMCPlots.plot_corner(post, true_theta=TRUTH, prior=prior, name="fig_corner_prior")
    MCMCPlots.configure(DERIVED_SYM, DERIVED_UNIT, np.ones(len(DERIVED_SYM)))
    MCMCPlots.plot_corner(D_post, true_theta=d_true, name="fig_corner_inertia")
    MCMCPlots.plot_corner(D_post, true_theta=d_true, prior=D_prior, name="fig_corner_inertia_prior")
    MCMCPlots.configure(PARAM_SYM, PARAM_UNIT, np.ones(len(PARAM_SYM)))
    plot_inertia_marginals(D_post, D_prior, d_true, d_hom)
    plot_core_sections(V, T, sub, TRUTH, D_post[:, 3:6], d_true[3:6])
    plot_residuals(model, sub, theta_hom)

    p_true = model.predict(TRUTH)
    write_summary(
        fig_dir / "summary.txt", D_post, D_prior, d_true, d_hom, post,
        [
            f"Shape {SHAPE_OBJ.name}: volume {outer.mass:.6e} m^3, R_v {r_v:.3f} m, "
            f"R_ref {R_REF} m, SH degree {N_DEG}",
            f"Observables: {n_obs}; pole {USE_POLE}, tumbling {USE_TUMBLING}",
            f"Truth principal moments A, B, C = {np.array2string(p_true['moments'], precision=6)} kg m^2",
            f"Bulk density {rho_bulk:.5f} g/cm^3; chi2 truth {np.sum(model.residuals(TRUTH) ** 2):.2f}, "
            f"homogeneous {r_hom @ r_hom:.1f}",
            f"MCMC: {N_WALKERS} walkers x {chain.shape[0]} steps, burn-in {burn}, thin {thin}, "
            f"tau max {np.max(tau):.1f}",
        ],
    )
    print(f"[Done] {time.time() - t_start:.0f} s; outputs in {fig_dir}")
