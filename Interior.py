r"""
Shared machinery of the interior-structure scripts (get_interior_density.py,
get_interior_inertia.py, get_interior_inertia_psyche.py, get_interior_inertia_justitia.py).

Shapes and mass integrals
-------------------------
load_shape, mesh_section, solid_harmonics, gauss01, fmt
    Triangle meshes (OBJ), surface cuts, fully normalized solid harmonics, Gauss-Legendre
    on [0, 1], summary-table numbers.
MeshShape, SHShape
    Star-shaped outer surfaces (triangle mesh or spherical-harmonic radius) with a common
    radius(u) interface.
AngularGrid
    Angular quadrature of a star-shaped body: every mass integral of a region {r < R(u)} is
    an angular integral with an analytic radial part (unit density),

        mass = int R^3 / 3 dOmega,  first = int u R^4 / 4 dOmega,  second = int u u^T R^5 / 5 dOmega.

Polyhedra: mass integrals and spherical-harmonic coefficients
-------------------------------------------------------------
mesh_volume, RadialLimit, fibonacci_sphere
    Signed volume, radial containment limit of a star-shaped mesh, quasi-uniform directions.
Moments, polyhedron_moments, BallRule, ellipsoid_moments
    Unit-density mass, first and second moments and unnormalized SH integrals of a polyhedron
    (exact Gauss quadrature on the origin-apex tetrahedra: tetra_gauss, face_tetrahedra,
    tetra_points) and of a triaxial ellipsoid.
central_inertia, inertia_tensor
    Centre of mass and inertia tensor about it; principal moments and axes.
sh_terms, cs_index, flatten_cs, cs_names, fully_normalized_legendre
    (n, m) orderings, coefficient vectors and labels, fully normalized Legendre functions.
tangent_basis, axis_offset
    Small-angle offsets of an observed axis (spin pole, long axis).

Two-layer moments of inertia (Ermakov 2026, PSJ 7:170)
-------------------------------------------------------
inertia_quantities, TwoLayer, endmember_map, masked_extreme, moment_grid_H_range
    Normalized principal moments, J2 = C_bar - (A_bar + B_bar) / 2, C22 = (B_bar - A_bar) / 4,
    H = J2 / C_bar, I_bar = (A_bar + B_bar + C_bar) / 3, for a core of radius r2 and density
    rho2 with boundary R_c(u) = min(R(u), r2 [(1 - w) + w R(u) / R_v]).
SolarPrecession
    Orbit-averaged solar precession of the spin pole (Ermakov 2026, Eq. 1).
MomentsPosterior, TwoLayerPosterior, LogCoreCoordinates
    MCMC (a) on (A_bar, B_bar, C_bar) and MCMC (b) on (r2, rho2, w), the latter sampled in
    phi = (ln r2, ln(rho2 - rho_mean), w).
initialize_walkers, multistart_walkers
    Least-squares walker starts (from the prior mean / from given multi-starts).
run_sampler, run_emcee, flat_posterior, with_mean_moment
    emcee with differential-evolution moves (optionally in a process pool), burn-in/thinning.
plot_endmembers, plot_sigma_I, plot_mean_moment, plot_traces, plot_core_sections
    Figures of the two-layer scripts (house style of MCMCPlots.py).
plot_trace_grid, plot_residual_band
    Walker traces on a 3-column grid; normalized residuals with a posterior predictive band.

Outputs
-------
results_dir, write_npz, write_text, quantiles, summary_row, ci_string
    results/<script>_<timestamp>/ directory, chain and summary files, summary numbers.
"""

from __future__ import annotations

import io
import math
from datetime import datetime
from math import factorial
from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.optimize import least_squares
from scipy.special import lpmv

import MCMCPlots

GM_SUN = 1.32712440018e20  # [m^3/s^2]
AU = 1.495978707e11  # [m]
OBLIQ_J2000 = 23.4392911  # obliquity of the ecliptic [deg]


# ============================================================
# Shapes and mass integrals
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


def mesh_section(V, T, axis):
    """
    Cut of the surface mesh by the plane x_axis = 0.

    Parameters
    ----------
    V : ndarray, shape (n_v, 3)
    T : ndarray of int, shape (n_t, 3)
    axis : int
        Normal of the cutting plane (0, 1, 2).

    Returns
    -------
    ndarray, shape (n, 2, 2)
        Segments, in the two other coordinates.
    """
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


def gauss01(n):
    """n-point Gauss-Legendre nodes and weights on [0, 1]."""
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def fmt(x):
    """Fixed-width number for the summary tables."""
    return f"{x:12.6g}"


class MeshShape:
    """
    Closed triangle mesh, star-shaped about the origin.

    Parameters
    ----------
    V : ndarray, shape (n_v, 3)
        Vertices [m].
    T : ndarray of int, shape (n_t, 3)
        Outward-oriented triangles.
    """

    def __init__(self, V, T):
        self.V, self.T = V, T

    @classmethod
    def from_obj(cls, path, unit=1.0):
        """Mesh of an OBJ file (load_shape); unit converts the OBJ length unit to metres."""
        return cls(*load_shape(path, unit))

    def integrals(self):
        """
        Exact unit-density mass integrals of the polyhedron (signed tetrahedra from the origin).

        Returns
        -------
        vol : float
            [m^3].
        first : ndarray, shape (3,)
            int x dV [m^4].
        second : ndarray, shape (3, 3)
            int x x^T dV [m^5].
        """
        V, F = self.V, self.T
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        det = np.einsum("ij,ij->i", a, np.cross(b, c))
        s = a + b + c
        second = sum(np.einsum("n,ni,nj->ij", det, x, x) for x in (a, b, c, s)) / 120.0
        return det.sum() / 6.0, det @ s / 24.0, second

    def radius(self, u, chunk=1024):
        """
        Surface radius along unit directions from the origin.

        Moller-Trumbore ray-triangle intersection of every ray with every face; the nearest
        hit in front of the origin is the surface.

        Parameters
        ----------
        u : ndarray, shape (N, 3)
            Unit directions.

        Returns
        -------
        ndarray, shape (N,)
            R(u) [m].
        """
        V, F = self.V, self.T
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


class SHShape:
    """
    Spherical-harmonic radius model R(u) = sum Pbar_nm(cos theta) [C_nm cos m phi + S_nm sin m phi].

    Parameters
    ----------
    C, S : ndarray, shape (L + 1, L + 1)
        4pi-normalized cosine and sine coefficients [m], indexed [n, m].
    """

    def __init__(self, C, S):
        self.C, self.S = C, S

    @classmethod
    def from_file(cls, path):
        """
        Read a text file with one "n, m, C_nm, S_nm" line per coefficient (4pi-normalized, m).
        """
        d = np.loadtxt(path, delimiter=",", ndmin=2)
        L = int(d[:, 0].max())
        C, S = np.zeros((L + 1, L + 1)), np.zeros((L + 1, L + 1))
        for n, m, c, s in d:
            C[int(n), int(m)], S[int(n), int(m)] = c, s
        return cls(C, S)

    @property
    def degree(self):
        return self.C.shape[0] - 1

    def radius(self, u):
        """
        Surface radius along unit directions in the shape frame.

        Parameters
        ----------
        u : ndarray, shape (N, 3)

        Returns
        -------
        ndarray, shape (N,)
            R(u) [m]. The Condon-Shortley phase of scipy's lpmv is removed.
        """
        C, S = self.C, self.S
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

    Gauss-Legendre in cos(theta) times a uniform rule in phi.

    Parameters
    ----------
    shape : MeshShape or SHShape
        Outer surface (any object with radius(u)), star-shaped about the origin.
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

    def __init__(self, shape, n_mu, n_phi):
        mu, wmu = np.polynomial.legendre.leggauss(n_mu)
        phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
        MU, PHI = (a.ravel() for a in np.meshgrid(mu, phi, indexing="ij"))
        st = np.sqrt(1.0 - MU**2)
        self.u = np.column_stack([st * np.cos(PHI), st * np.sin(PHI), MU])
        self.w = np.outer(wmu, np.full(n_phi, 2.0 * np.pi / n_phi)).ravel()
        self._uu = np.einsum("ni,nj->nij", self.u, self.u).reshape(-1, 9)
        self.R = shape.radius(self.u)
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


# ============================================================
# Polyhedra: mass integrals and spherical-harmonic coefficients
# ============================================================


def mesh_volume(V, T):
    """Signed volume of a closed triangle mesh (origin-apex tetrahedra), > 0 if outward-oriented."""
    v0 = V[T[:, 0]]
    v1 = V[T[:, 1]]
    v2 = V[T[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0)


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


def tetra_gauss(deg):
    """
    Gauss rules in (t, s, q) for polynomials of degree <= deg on the origin-apex tetrahedra.

    x = t [(1 - s) v0 + s (1 - q) v1 + s q v2] maps the unit cube onto the tetrahedron
    (0, v0, v1, v2) with dV = 6 vol t^2 s dt ds dq (Duffy), so the integrand is of degree
    deg + 2 in t, deg + 1 in s and deg in q.

    Returns
    -------
    (t, wt), (s, ws), (q, wq) : gauss01 rules
    """
    return gauss01((deg + 4) // 2), gauss01((deg + 3) // 2), gauss01((deg + 2) // 2)


def face_tetrahedra(V, T, chunk):
    """
    Batches of the origin-apex tetrahedra (0, v0, v1, v2) of the faces.

    Yields
    ------
    v0, v1, v2 : ndarray, shape (n, 3)
        Face vertices (n <= chunk).
    vol : ndarray, shape (n,)
        Signed tetrahedron volumes.
    """
    for i in range(0, len(T), chunk):
        v0, v1, v2 = (V[T[i : i + chunk, k]] for k in range(3))
        yield v0, v1, v2, np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0


def tetra_points(c0, c1, c2, v0, v1, v2):
    """Points c0 v0 + c1 v1 + c2 v2 of every face and node, shape (n_faces * n_nodes, 3)."""
    return (
        c0[None, :, None] * v0[:, None, :]
        + c1[None, :, None] * v1[:, None, :]
        + c2[None, :, None] * v2[:, None, :]
    ).reshape(-1, 3)


def polyhedron_moments(V, T, n_max, r_ref, chunk=4000):
    """
    Unit-density mass integrals of a closed polyhedron, exact up to degree n_max.

    The body is split into the signed tetrahedra (origin, v0, v1, v2) of its faces, each
    integrated with the tensor Gauss rule of tetra_gauss.

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
    (t, wt), (s, ws), (q, wq) = tetra_gauss(max(n_max, 2))
    TT, SS, QQ = (g.ravel() for g in np.meshgrid(t, s, q, indexing="ij"))
    W = (np.einsum("i,j,k->ijk", wt * t**2, ws * s, wq)).ravel() * 6.0
    c0, c1, c2 = TT * (1.0 - SS), TT * SS * (1.0 - QQ), TT * SS * QQ  # barycentric

    mass, first, second = 0.0, np.zeros(3), np.zeros((3, 3))
    Ic = np.zeros((n_max + 1, n_max + 1))
    Is = np.zeros_like(Ic)
    for v0, v1, v2, vol in face_tetrahedra(V, T, chunk):
        P = tetra_points(c0, c1, c2, v0, v1, v2)
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
        r, wr = gauss01(n_max // 2 + 3)
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


def central_inertia(mass, first, second):
    """
    Centre of mass [m] and inertia tensor about it, from int dm, int x dm, int x x^T dm.

    Returns
    -------
    com : ndarray, shape (3,)
    I : ndarray, shape (3, 3)
    """
    com = first / mass
    Sc = second - mass * np.outer(com, com)
    return com, np.trace(Sc) * np.eye(3) - Sc


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
    com, I = central_inertia(mom.mass, mom.first, mom.second)
    moments, axes = np.linalg.eigh(I)
    return com, I, moments, axes


def sh_terms(n_max, exclude_c00=True):
    """(n, m) for n = 0..n_max, m = 0..n, without (0, 0) if exclude_c00."""
    return [(n, m) for n in range(n_max + 1) for m in range(n + 1)
            if not (exclude_c00 and n == 0 and m == 0)]


def cs_index(n_max):
    """(kind, n, m) of the observed coefficients: degree 1..n_max, C_nm for m >= 0, S_nm for m >= 1."""
    out = []
    for n in range(1, n_max + 1):
        for m in range(n + 1):
            out.append(("C", n, m))
            if m > 0:
                out.append(("S", n, m))
    return out


def flatten_cs(C, S, n_degree, exclude_c00=True):
    """
    Stack the coefficients as [C_nm ..., S_nm ...] in sh_terms order.

    Parameters
    ----------
    C, S : ndarray, shape (>= n_degree + 1, >= n_degree + 1)
    n_degree : int
    exclude_c00 : bool
        Skip (n, m) = (0, 0) in both blocks.

    Returns
    -------
    ndarray, shape (2 n_terms,)
    """
    nm = sh_terms(n_degree, exclude_c00)
    return np.array([C[n, m] for n, m in nm] + [S[n, m] for n, m in nm], dtype=float)


def cs_names(n_degree, exclude_c00=True):
    """Labels of the entries of flatten_cs, e.g. 'C20', 'S22'."""
    nm = sh_terms(n_degree, exclude_c00)
    return [f"C{n}{m}" for n, m in nm] + [f"S{n}{m}" for n, m in nm]


def fully_normalized_legendre(nmax, x):
    """
    Fully normalized associated Legendre functions Pbar_nm(x), x = sin(latitude).

    Normalization: the integral of Pbar_nm^2 [cos^2, sin^2](m lam) over the unit sphere is
    4 pi, so Pbar_10 = sqrt(3) x and Pbar_11 = sqrt(3) sqrt(1 - x^2). Standard
    sectoral / subdiagonal / column recursions.

    Parameters
    ----------
    nmax : int
        Maximum degree.
    x : float
        sin(latitude), in [-1, 1].

    Returns
    -------
    ndarray, shape (nmax + 1, nmax + 1)
        P[n, m] (zero for m > n).
    """
    P = np.zeros((nmax + 1, nmax + 1), dtype=float)
    P[0, 0] = 1.0
    if nmax == 0:
        return P

    sx = math.sqrt(max(0.0, 1.0 - x * x))
    for m in range(1, nmax + 1):
        # Pbar_11 = sqrt(3) cos(phi); for m >= 2 the delta_m0 normalization factor is constant
        f = math.sqrt(3.0) if m == 1 else math.sqrt((2.0 * m + 1.0) / (2.0 * m))
        P[m, m] = f * sx * P[m - 1, m - 1]

    for m in range(0, nmax):
        P[m + 1, m] = math.sqrt(2.0 * m + 3.0) * x * P[m, m]

    for m in range(0, nmax + 1):
        for n in range(m + 2, nmax + 1):
            a = math.sqrt(((2.0 * n + 1.0) * (2.0 * n - 1.0)) / ((n - m) * (n + m)))
            b = math.sqrt(
                ((2.0 * n + 1.0) * (n + m - 1.0) * (n - m - 1.0))
                / ((2.0 * n - 3.0) * (n - m) * (n + m))
            )
            P[n, m] = a * x * P[n - 1, m] - b * P[n - 2, m]
    return P


def tangent_basis(u):
    """Two unit vectors orthogonal to the unit vector u (and to each other)."""
    a = np.eye(3)[np.argmin(np.abs(u))]
    t1 = np.cross(u, a)
    t1 /= np.linalg.norm(t1)
    return t1, np.cross(u, t1)


def axis_offset(e, u_obs, t1, t2):
    """Small-angle offset [rad] of the axis e (sign-free) from u_obs, along t1 and t2."""
    e = e if e @ u_obs >= 0.0 else -e
    return np.array([e @ t1, e @ t2])


# ============================================================
# Two-layer moments of inertia
# ============================================================


def inertia_quantities(mass, first, second, r0, axes=False):
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
    r0 : float
        Normalization radius [m].
    axes : bool
        Also return the principal axes (eigh instead of eigvalsh).

    Returns
    -------
    dict
        A, B, C (normalized by M r0^2, ascending), J2, C22, H, I (mean moment), com [m],
        and with axes=True, axes (principal axes as columns, in the order A, B, C).
    """
    com, Ic = central_inertia(mass, first, second)
    if axes:
        lam, vec = np.linalg.eigh(Ic)
    else:
        lam = np.linalg.eigvalsh(Ic)
    a, b, c = lam / (mass * r0**2)
    J2 = c - 0.5 * (a + b)
    out = dict(A=a, B=b, C=c, J2=J2, C22=0.25 * (b - a), H=J2 / c, I=(a + b + c) / 3.0, com=com)
    if axes:
        out["axes"] = vec
    return out


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
    r0 : float
        Normalization radius of the moments [m].
    axes : bool
        Principal axes in the output (inertia_quantities with axes=True).
    """

    def __init__(self, grid, rho_mean, r0, axes=False):
        self.g, self.rho_mean, self.r0, self.axes = grid, rho_mean, r0, axes

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
            rho1 * V + (rho2 - rho1) * v, rho1 * F + (rho2 - rho1) * f, rho1 * S2 + (rho2 - rho1) * s,
            self.r0, axes=self.axes,
        )
        d["rho1"] = rho1
        d["r_c"] = (3.0 * v / (4.0 * np.pi)) ** (1.0 / 3.0) / 1e3
        return d


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


def moment_grid_H_range(hom, frac, n=8):
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


class SolarPrecession:
    """
    Orbit-averaged solar precession of the spin pole (Ermakov 2026, Eq. 1).

    dp/dt = K H (k . p) (p x k), K = 3 n^2 / (2 omega (1 - e^2)^{3/2}): the pole p circles the
    orbit normal k at the rate Omega_p = K H cos(eps).

    Parameters
    ----------
    pole_ra, pole_dec : float
        Spin pole, ICRF [deg].
    spin_rate : float
        Rotation rate [deg/day].
    orbit : dict
        Heliocentric a [AU], e, i [deg], node [deg] (ecliptic J2000).
    """

    def __init__(self, pole_ra, pole_dec, spin_rate, orbit):
        self.pole_ra, self.pole_dec, self.spin_rate, self.orbit = pole_ra, pole_dec, spin_rate, orbit

    def __call__(self, H):
        """
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
        n2 = GM_SUN / (self.orbit["a"] * AU) ** 3
        omega = np.radians(self.spin_rate) / 86400.0
        K = 1.5 * n2 / omega / (1.0 - self.orbit["e"] ** 2) ** 1.5 * yr  # [rad/yr]
        p = unit_radec(self.pole_ra, self.pole_dec)
        k = orbit_normal_icrf(self.orbit["i"], self.orbit["node"])
        cos_eps = p @ k
        dp = np.multiply.outer(K * H * cos_eps, np.cross(p, k))  # [rad/yr]
        a, d = np.radians(self.pole_ra), np.radians(self.pole_dec)
        mas = np.degrees(1.0) * 3.6e6
        dra = (-np.sin(a) * dp[..., 0] + np.cos(a) * dp[..., 1]) / np.cos(d) * mas
        ddec = dp[..., 2] / np.cos(d) * mas
        omega_p = K * H * cos_eps
        return dict(
            eps=np.degrees(np.arccos(cos_eps)), omega_p=omega_p, period=2 * np.pi / np.abs(omega_p) / 1e6,
            dra=dra, ddec=ddec, rate=np.linalg.norm(dp, axis=-1) * mas,
        )


# ============================================================
# Posteriors and sampling
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

    Prior: uniform box lo..hi with rho1 >= rho1_min.

    Parameters
    ----------
    model : TwoLayer
    obs, sigma : dict
        Observed J2, C22, H and their standard deviations.
    lo, hi : ndarray, shape (3,)
        Prior box.
    rho1_min : float
        Minimum mantle density [kg/m^3].
    """

    keys = ("J2", "C22", "H")
    derived_names = ("I_bar", "C_bar", "H", "J2", "rho1", "r_c")

    def __init__(self, model, obs, sigma, lo, hi, rho1_min):
        self.model, self.lo, self.hi = model, np.asarray(lo), np.asarray(hi)
        self.rho1_min = rho1_min
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
        return self.model(*theta)["rho1"] >= self.rho1_min

    def __call__(self, theta):
        if np.any(theta < self.lo) or np.any(theta > self.hi):
            return -np.inf
        d = self.predict(theta)
        if not d["rho1"] >= self.rho1_min:
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
        """derived_names of one sample."""
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
    rho_mean : float, optional
        Default: the mean density of post.model.
    """

    def __init__(self, post, rho_mean=None):
        self.post = post
        self.rho_mean = post.model.rho_mean if rho_mean is None else rho_mean

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

    Parameters
    ----------
    post : MomentsPosterior or TwoLayerPosterior
        Needs lo, hi, residuals, sample_prior and __call__.

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


def multistart_walkers(post, starts, n_walkers, rng, lo, hi, select="cost", max_nfev=400,
                       max_draws=None):
    """
    Multi-start bounded least squares, then walkers around the best fit.

    Walkers are drawn from N(theta_map, cov), cov = (J^T J)^-1 at the best fit (a small
    diagonal 1e-4 (hi - lo) if J^T J is singular), and redrawn until they have finite
    posterior.

    Parameters
    ----------
    post : callable
        Log posterior, with a residuals(theta) method (normalized residuals).
    starts : iterable of ndarray, shape (ndim,)
        Least-squares starting points (e.g. prior draws).
    n_walkers : int
    rng : numpy.random.Generator
        For the walker draws.
    lo, hi : ndarray, shape (ndim,)
        Bounds of the fits (prior box).
    select : {"cost", "logp"}
        Best fit: smallest chi2, or largest log posterior (a fit outside the prior support,
        log p = -inf, is never selected).
    max_nfev : int
        Function evaluations per fit.
    max_draws : int, optional
        Walker draws before falling back to the small diagonal (None: no limit).

    Returns
    -------
    p0 : ndarray, shape (n_walkers, ndim)
    theta_map : ndarray, shape (ndim,)
        Best fit.
    cov : ndarray, shape (ndim, ndim) or None
        Its covariance (None if singular).

    Raises
    ------
    RuntimeError
        If select="logp" and no fit has finite log posterior.
    """
    starts = list(starts)
    width = hi - lo
    best, best_lp = None, -np.inf
    for k, x0 in enumerate(starts):
        sol = least_squares(
            post.residuals, x0, bounds=(lo, hi), x_scale=width, method="trf",
            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=max_nfev,
        )
        lp = post(sol.x)
        print(f"[Init] start {k + 1:2d}/{len(starts)}: chi2 = {2 * sol.cost:14.4f}  "
              f"log p = {lp:14.4f}")
        if select == "logp":
            if lp > best_lp:
                best, best_lp = sol, lp
        elif best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        raise RuntimeError("no least-squares start reached the prior support; widen the search")
    try:
        cov = np.linalg.inv(best.jac.T @ best.jac)
        L = np.linalg.cholesky(0.5 * (cov + cov.T))
    except np.linalg.LinAlgError:
        cov, L = None, np.diag(1e-4 * width)
    p0, draws = [], 0
    while len(p0) < n_walkers:
        draws += 1
        if max_draws is not None and draws > max_draws:
            L = np.diag(1e-4 * width)
        th = best.x + L @ rng.standard_normal(best.x.size)
        if np.isfinite(post(th)):
            p0.append(th)
    print(f"[Init] best fit: chi2 = {2 * best.cost:.4f} ({best.fun.size} observables, "
          f"{best.x.size} parameters)")
    return np.array(p0), best.x, cov


def run_sampler(target, p0, n_steps, n_proc=1, progress=True):
    """
    emcee with differential-evolution moves (DEMove 0.8, DESnookerMove 0.2).

    Parameters
    ----------
    target : callable
        Log posterior (picklable if n_proc > 1).
    p0 : ndarray, shape (n_walkers, ndim)
        Walker starts.
    n_steps : int
    n_proc : int
        > 1: multiprocessing pool of that size.
    progress : bool
        emcee progress bar.

    Returns
    -------
    chain : ndarray, shape (n_steps, n_walkers, ndim)
    logp : ndarray, shape (n_steps, n_walkers)
    tau : ndarray, shape (ndim,)
        Autocorrelation times (tol=0).
    acceptance : float
        Mean acceptance fraction.
    """
    n_walkers, ndim = p0.shape
    moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    if n_proc > 1:
        from multiprocessing import Pool

        with Pool(n_proc) as pool:
            sampler = emcee.EnsembleSampler(n_walkers, ndim, target, moves=moves, pool=pool)
            sampler.run_mcmc(p0, n_steps, progress=progress)
    else:
        sampler = emcee.EnsembleSampler(n_walkers, ndim, target, moves=moves)
        sampler.run_mcmc(p0, n_steps, progress=progress)
    tau = sampler.get_autocorr_time(tol=0)
    return (sampler.get_chain(), sampler.get_log_prob(), tau,
            float(np.mean(sampler.acceptance_fraction)))


def run_emcee(post, n_walkers, n_steps, rng, label, coords=None):
    """
    emcee with differential-evolution moves (DEMove 0.8, DESnookerMove 0.2).

    Parameters
    ----------
    post : MomentsPosterior or TwoLayerPosterior
    n_walkers, n_steps : int
    rng : numpy.random.Generator
        For the walker starts.
    label : str
        Printed with the acceptance and autocorrelation time.
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
    p0, theta_map = initialize_walkers(post, n_walkers, rng)
    target = post if coords is None else coords
    if coords is not None:
        p0 = coords.to_phi(p0)
    chain, logp, tau, acc = run_sampler(target, p0, n_steps, progress=False)
    print(f"[MCMC] {label}: acceptance {acc:.3f}, "
          f"tau max {np.max(tau):.1f} ({n_steps / np.max(tau):.0f} tau)")
    if coords is not None:
        logp = logp - coords.log_jacobian(chain)
        chain = coords.to_theta(chain)
    return chain, logp, tau, theta_map


def flat_posterior(chain, tau, burn_in=None, thin=None):
    """
    Flattened chain after burn-in and thinning.

    Parameters
    ----------
    burn_in, thin : int, optional
        Default: 5 x max tau and max(1, min tau / 2), as in get_interior_inertia.py.

    Returns
    -------
    samples : ndarray, shape (n, ndim)
    burn, thin : int
    """
    burn = int(5 * np.max(tau)) if burn_in is None else int(burn_in)
    thin = max(1, int(0.5 * np.min(tau))) if thin is None else int(thin)
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


def plot_endmembers(r2s, rho2s, maps, r_min, extremes, rho1_min, rho_mean, levels=None, fmt_J2="%.3f",
                    name="fig_endmembers"):
    """
    As Ermakov (2026, Fig. 2): H (black), J2 (vermillion) and rho1 (grey dashed) over core
    radius and density for the spherical (A) and rescaled-shape (B) cores; rho1 < rho1_min
    shaded.

    Parameters
    ----------
    r2s, rho2s : ndarray
        Grid of core radius [km] and density [kg/m^3].
    maps : list of dict
        endmember_map for w = 0 and w = 1.
    r_min : float
        Smallest surface radius [km].
    extremes : list of (value, (i, j))
        Marked grid point per panel (masked_extreme).
    rho1_min, rho_mean : float
        Minimum mantle density and mean density [kg/m^3].
    levels : dict, optional
        Contour levels for keys "H", "J2", "rho1"; missing keys are chosen from the admissible
        values and shared by the two panels.
    fmt_J2 : str
        Label format of the J2 contours.
    """
    levels = {} if levels is None else levels

    def auto(key, n):
        vals = np.concatenate([np.where(m["rho1"] >= rho1_min, m[key], np.nan).ravel() for m in maps])
        return MaxNLocator(n).tick_values(np.nanmin(vals), np.nanmax(vals))[1:-1]

    lev_H = levels["H"] if "H" in levels else auto("H", 7)
    lev_J2 = levels["J2"] if "J2" in levels else auto("J2", 6)
    lev_rho1 = levels["rho1"] if "rho1" in levels else MaxNLocator(4).tick_values(rho1_min, rho_mean)[1:-1]
    fig, axs = plt.subplots(1, 2, figsize=(MCMCPlots.TXT_W, 2.9), sharey=True, layout="constrained")
    X, Y = np.meshgrid(r2s, rho2s / 1e3, indexing="ij")
    for ax, m, title, ext in zip(axs, maps, ("(A) Spherical Core", "(B) Rescaled-Shape Core"), extremes):
        rho1 = np.nan_to_num(m["rho1"], nan=-1e5)  # nan: core fills the body
        ax.contourf(X, Y, rho1, levels=[rho1.min() - 1.0, rho1_min], colors=[MCMCPlots.C_SPAN], zorder=0)
        keep = m["rho1"] >= 0.9 * rho1_min  # H, J2 only where the mantle is admissible (with margin)
        H, J2 = np.where(keep, m["H"], np.nan), np.where(keep, m["J2"], np.nan)
        if np.nanmax(H) - np.nanmin(H) > 1e-3:
            cs = ax.contour(X, Y, H, levels=lev_H, colors="k", linewidths=0.7)
            ax.clabel(cs, fontsize=6, fmt="%.2f", inline_spacing=2)
        else:  # rescaled-shape core: H does not depend on the core (Ermakov 2026, Sec. 4.1)
            ax.text(0.04, 0.05, rf"$H = {np.nanmean(H):.4f}$ everywhere", transform=ax.transAxes,
                    fontsize=7, bbox=dict(fc="white", ec="none", pad=1.5))
        cj = ax.contour(X, Y, J2, levels=lev_J2, colors=MCMCPlots.C_BATCH, linewidths=0.7)
        ax.clabel(cj, fontsize=6, fmt=fmt_J2, inline_spacing=2)
        cr = ax.contour(X, Y, m["rho1"], levels=lev_rho1, colors="0.55", linewidths=0.6, linestyles="--")
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
        Patch(color=MCMCPlots.C_SPAN, label=rf"$\rho_1 < {rho1_min:.0f}$ kg/m$^3$"),
        Line2D([], [], ls="", marker="*", ms=8, mfc=MCMCPlots.C_POST, mec="white",
               label=r"Minimum $\bar{I}$ ($\rho_1 \geq %.0f$)" % rho1_min),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=5, frameon=False)
    return MCMCPlots.save_figure(fig, name)


def plot_sigma_I(sweep, sig_I, sig_I_noH, C_true, C_hom, prior_frac, paper_sigma_H=None, name="fig_sigma_I"):
    """
    Posterior standard deviation of I_bar against sigma_H / H (MCMC (a)), with Eq. (13)
    for the truth C_bar and the prior range of C_bar, and the sigma_I = 0.01 requirement.

    Parameters
    ----------
    prior_frac : (float, float)
        Prior range of the moments as fractions of homogeneous.
    paper_sigma_H : sequence of float, optional
        Published sigma_H / H [%] for sigma_I = 0.01, marked on the requirement line.
    """
    x = np.linspace(0.0, 1.05 * max(sweep), 200)
    fig, ax = plt.subplots(figsize=(MCMCPlots.COL_W, 2.5), layout="constrained")
    ax.fill_between(100 * x, prior_frac[0] * C_hom * x, prior_frac[1] * C_hom * x,
                    color=MCMCPlots._tint(MCMCPlots.C_POST, 0.15), lw=0,
                    label=r"Eq. (13), $\bar{C}$ at 75--110$\%$ of Homogeneous")
    ax.plot(100 * x, C_true * x, color=MCMCPlots.C_POST, lw=0.9, label=r"Eq. (13), True $\bar{C}$")
    ax.plot(100 * np.array(sweep), sig_I, "o", ms=3.5, color=MCMCPlots.C_POST, label="MCMC (a)")
    ax.axhline(sig_I_noH, color=MCMCPlots.C_PRIOR, ls="--", lw=0.9, label=r"MCMC (a) without $H$")
    ax.axhline(0.01, color=MCMCPlots.C_TRUTH, ls=":", lw=0.8)
    if paper_sigma_H is None:
        ax.text(0.98, 0.01, r"$\sigma_{\bar{I}} = 0.01$", transform=ax.get_yaxis_transform(), ha="right",
                va="bottom", fontsize=6.5, color=MCMCPlots.C_TRUTH)
    else:
        for s in paper_sigma_H:
            ax.plot(s, 0.01, "D", ms=3.5, mfc="white", mec=MCMCPlots.C_BATCH, mew=0.9)
        ax.plot([], [], "D", ms=3.5, mfc="white", mec=MCMCPlots.C_BATCH,
                label=r"Paper, $\sigma_{\bar{I}} = 0.01$")
    ax.set_xlabel(r"$\sigma_H / H$ [$\%$]")
    ax.set_ylabel(r"$\sigma_{\bar{I}}$ [-]")
    ax.set_xlim(0, 100 * x[-1])
    ax.set_ylim(0, 1.15 * max(np.max(sig_I), sig_I_noH))
    fig.legend(loc="outside upper center", ncol=2, frameon=False, fontsize=6.5)
    return MCMCPlots.save_figure(fig, name)


def plot_mean_moment(curves, I_true, I_hom, I_min, min_label=r"Minimum, Endmember (A)",
                     name="fig_mean_moment"):
    """
    Marginals of I_bar: prior (a), MCMC (a) without and with H, MCMC (b), against the truth,
    the homogeneous body and a lower bound I_min.

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
    ax.axvline(I_min, color="0.45", ls="-.", lw=0.9, label=min_label)
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


def plot_trace_grid(chain, logp, burn, ref=None, ref_color=None, scale=None, name="fig_traces"):
    """
    Walker traces of every parameter and of log p on a 3-column grid (burn-in shaded).

    Parameters
    ----------
    chain : ndarray, shape (n_step, n_walk, n_par)
    logp : ndarray, shape (n_step, n_walk)
    burn : int
        Burn-in steps (shaded).
    ref : ndarray, shape (n_par,), optional
        Reference values (dashed), e.g. the truth or the least-squares fit.
    ref_color : str, optional
        Their colour (default MCMCPlots.C_TRUTH).
    scale : ndarray, shape (n_par,), optional
        Display scale of each parameter (e.g. MCMCPlots.PARAM_SCALE).
    name : str
        File name of the PDF.

    Returns
    -------
    Path or None
        The PDF written.
    """
    n_step, n_walk, n_par = chain.shape
    s = np.ones(n_par) if scale is None else scale
    ref_color = MCMCPlots.C_TRUTH if ref_color is None else ref_color
    stride = max(1, n_step // 2000)
    it = np.arange(0, n_step, stride)
    ncol = 3
    nrow = int(np.ceil((n_par + 1) / ncol))
    fig, axs = plt.subplots(
        nrow, ncol, figsize=(MCMCPlots.TXT_W, 1.15 * nrow), sharex=True, layout="constrained"
    )
    axs = axs.ravel()
    series = [chain[..., i] * s[i] for i in range(n_par)] + [logp]
    labels = [MCMCPlots.param_label(i) for i in range(n_par)] + [r"$\log p$"]
    for i, (ax, y) in enumerate(zip(axs, series)):
        ax.axvspan(0, burn, color=MCMCPlots.C_SPAN, lw=0, zorder=0)
        ax.plot(it, y[it], color=MCMCPlots.C_POST, lw=0.3, alpha=0.25, rasterized=True)
        if i < n_par:
            if ref is not None:
                ax.axhline(ref[i] * s[i], color=ref_color, ls="--", lw=0.8)
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


def plot_residual_band(model, post, names, refs=(), keep=None, name="fig_residuals", n_draw=300):
    """
    Normalized residuals of every observable: posterior predictive band (median, 95 %) and
    reference solutions (+/-3 sigma shaded; symlog scale beyond 10 sigma).

    Parameters
    ----------
    model : object
        With residuals(theta) (normalized residuals).
    post : ndarray, shape (N, ndim)
        Posterior samples.
    names : list of str
        Observable labels (residuals order).
    refs : iterable of (theta, marker, ms, color, label)
        Reference solutions (e.g. least squares, homogeneous body).
    keep : ndarray of bool, optional
        Observables to show (default all).
    name : str
    n_draw : int
        Posterior draws for the predictive band.

    Returns
    -------
    Path or None
    """
    keep = np.ones(len(names), dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    rng = np.random.default_rng(2)
    draws = post[rng.choice(len(post), size=min(n_draw, len(post)), replace=False)]
    R = np.array([model.residuals(th) for th in draws])[:, keep]
    names = [nm for nm, k_ in zip(names, keep) if k_]
    k = np.arange(len(names))
    lo, med, hi = np.percentile(R, [2.5, 50, 97.5], axis=0)
    fig, ax = plt.subplots(figsize=(MCMCPlots.TXT_W, 2.4), layout="constrained")
    ax.axhspan(-3, 3, color=MCMCPlots.C_SPAN, lw=0, zorder=0)
    ax.vlines(k, lo, hi, color=MCMCPlots.C_POST, lw=1.6, alpha=0.5)
    ax.plot(k, med, "o", ms=2.4, color=MCMCPlots.C_POST, label=r"MCMC Posterior (median, 95$\%$)")
    r_max = max(np.abs(lo).max(), np.abs(hi).max())
    for theta, marker, ms, color, label in refs:
        r = model.residuals(theta)[keep]
        r_max = max(r_max, np.abs(r).max())
        ax.plot(k, r, marker, ls="", ms=ms, mew=0.8, color=color, label=label)
    if r_max > 10.0:
        ax.set_yscale("symlog", linthresh=3.0)
    fs = 6 if len(k) <= 60 else 4
    ax.set_xticks(k, names, rotation=90, fontsize=fs)
    ax.set_xlim(-0.7, k[-1] + 0.7)
    ax.set_ylabel(r"(Observed $-$ Model) / $\sigma$")
    fig.legend(loc="outside upper center", ncol=1 + len(refs))
    return MCMCPlots.save_figure(fig, name)


def plot_core_sections(shape, model, post, truth, surface_label, name="fig_core_sections", n_draw=80):
    """
    Outer shape and core boundaries in the x-z (meridional) and x-y (equatorial) planes of
    the body frame: truth (dashed), posterior draws of MCMC (b) (blue), the r_min sphere.

    Parameters
    ----------
    shape : MeshShape or SHShape
    model : TwoLayer
    post : ndarray, shape (n, 3)
        Posterior samples of (r2 [km], rho2, w).
    truth : ndarray, shape (3,)
    surface_label : str
        Legend entry of the surface.
    """
    t = np.linspace(0.0, 2.0 * np.pi, 721)
    planes = [("x", "z", np.column_stack([np.cos(t), np.zeros_like(t), np.sin(t)]), (0, 2)),
              ("x", "y", np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)]), (0, 1))]
    fig, axs = plt.subplots(1, 2, figsize=(MCMCPlots.TXT_W, 2.9), layout="constrained")
    r_min = model.g.R.min()
    for ax, (lx, ly, u, (i, j)) in zip(axs, planes):
        R = shape.radius(u)
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
        Line2D([], [], color="0.25", lw=0.9, label=surface_label),
        Line2D([], [], color=MCMCPlots.C_TRUTH, ls="--", lw=1.0, label="Core (Truth)"),
        Line2D([], [], color=MCMCPlots.C_POST, lw=0.8, label="Core (Posterior Draws)"),
        Line2D([], [], color="0.45", ls=":", lw=0.7, label=r"$r_{\min}$ Sphere"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False)
    return MCMCPlots.save_figure(fig, name)


# ============================================================
# Outputs
# ============================================================


def results_dir(script_file):
    """
    Create results/<script name>_<YYMMDDHHMMSS>/ next to the script and make it the figure
    directory of MCMCPlots.

    Parameters
    ----------
    script_file : str or Path
        __file__ of the calling script.

    Returns
    -------
    Path
    """
    script = Path(script_file).resolve()
    return MCMCPlots.set_figure_dir(
        script.parent / "results" / (script.stem + "_" + datetime.now().strftime("%y%m%d%H%M%S"))
    )


def write_npz(path, **arrays):
    """Compressed .npz of the arrays (np.savez_compressed), written with MCMCPlots.write_output."""
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return MCMCPlots.write_output(path, buf.getvalue(), tag="Chain")


def write_text(path, lines):
    """Print the lines and write them to path (MCMCPlots.write_output)."""
    text = "\n".join(lines) + "\n"
    print(text)
    return MCMCPlots.write_output(path, text.encode(), tag="Summary")


def quantiles(x):
    """Median and 16-84 % half-widths of samples: (median, median - q16, q84 - median)."""
    q16, q50, q84 = np.percentile(x, [16, 50, 84])
    return q50, q50 - q16, q84 - q50


def summary_row(name, ref, ours, fmt="{:.4f}"):
    """Summary line: quantity, reference value, our value (strings kept as they are)."""
    r = ref if isinstance(ref, str) else fmt.format(ref)
    o = ours if isinstance(ours, str) else fmt.format(ours)
    return f"  {name:<46s} {r:>18s} {o:>22s}"


def ci_string(x):
    """Median and 16-84 % interval of samples, as 'med +up -down'."""
    lo, med, hi = np.percentile(x, [16, 50, 84])
    return f"{med:.4f} +{hi - med:.4f} -{med - lo:.4f}"
