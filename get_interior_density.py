r"""
Interior density of an asteroid from its spherical-harmonic gravity field, by density
regions (or mascons) and MCMC (emcee).

No orbit propagation, no MCMCModel. The OBJ loader, solid harmonics, Gauss rules and mesh
cuts come from Interior.py; the figures reuse the house style of MCMCPlots.py (configure,
plot_corner, save_figure, write_output).

Model
-----
The body (polyhedron SHAPE_OBJ: the OSIRIS-REx shape model of Bennu, in the frame of its
gravity field) is split into K density groups of uniform density; one density per group is
estimated:

  theta = [rho_0, ..., rho_{K-1}]       [kg/m^3] (displayed in g/cm^3)

Every coefficient is a ratio of mass integrals, linear in the densities:

  Cbar_nm + i Sbar_nm = sum_k rho_k I_k,nm / ((2n + 1) sum_k rho_k V_k),
  I_k,nm = int_{group k} (r / R)^n Pbar_nm(sin phi) e^{i m lam} dV,

so the model is y(theta) = A w / sum(w), w_k = rho_k V_k, with the design matrix A
(columns I_k / ((2n + 1) V_k)) built once. Group layouts (MASCON_MODE):
  * "regions" (default): 7 regions, integrated exactly. A central ball (radius
    CORE_RADIUS) and, outside it, an equatorial belt (|latitude| < BELT_LAT_DEG) split in
    four 90-deg longitude quadrants, plus the north and south caps. The integrals use the
    origin-apex tetrahedra of the mesh with a Duffy Gauss rule, each ray split at the ball
    radius: exact for the homogeneous body (no discretization error) and to the face size
    on the region boundaries.
  * "strategic": K point masses of equal volume (V_body / K) on the principal axes of the
    vertex cloud (centre, +/- long, +/- intermediate, +/- pole). Crude: even with equal
    densities it does not reproduce the field of the homogeneous body.
  * "layered": N_MASCONS random point masses uniform in volume, in N_LAYERS radial shells
    of equal thickness (Monte Carlo noise ~ 1 / sqrt(N_MASCONS) on every coefficient).

Observables
-----------
  * Cbar_nm, Sbar_nm for n = 1..N_DEG (C_00 excluded) of the literature field of Bennu
    (bennu_truth_cs_from_literature). Degree 1 is the centre-of-mass offset
    (C_10 = z_cm / (sqrt(3) R), ...), sigma from SIGMA_COM; degree n >= 2 has the data
    sigma SIGMA_CS[n] plus the representation error SIGMA_MODEL in quadrature. A 7-region
    model cannot reproduce the field to its formal accuracy (degree 4: 24 informative
    coefficients + GM; best fit chi2 ~ 270 with SIGMA_MODEL = 0, the homogeneous body
    ~ 2100); SIGMA_MODEL = 1.6e-4 brings the best fit to chi2 ~ 19 for 18 degrees of
    freedom. S_n0 = 0 identically and carries no information.
  * USE_GM: GM = G sum(w) with sigma SIGMA_GM. Without it the coefficients are invariant
    to a common scaling of all densities and the posterior is a ridge along that scaling,
    bounded only by the prior box.
Prior: independent uniform densities in [RHO_LO, RHO_HI].

Sampler (as get_interior_inertia.py): multi-start bounded least squares from prior draws,
walkers around the best fit, emcee with DEMove (0.8) and DESnookerMove (0.2), burn-in
5 tau and thinning tau / 2.

Outputs (results/<script name>_<YYMMDDHHMMSS>/)
-----------------------------------------------
chain.npz (full chain, log-probabilities, design matrix, data, settings; replot with
LOAD_CHAIN), summary.txt (posterior, least squares and homogeneous body), and one PDF per
figure: traces, density corner (+ prior), densities per group, density sections through
the body, normalized residuals.
"""

from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import uniform

import MCMCPlots
from Interior import fmt, gauss01, load_shape, mesh_section, solid_harmonics

MCMCPlots.SHOW_FIGS = True  # plt.show() after each figure is saved

# Optional (fast contains test + volume sampling). If unavailable, we fall back.
_TRIMESH_AVAILABLE = False
try:
    import trimesh  # pip install trimesh

    _TRIMESH_AVAILABLE = True
except Exception:
    _TRIMESH_AVAILABLE = False

HERE = Path(__file__).resolve().parent

# ============================================================
# Settings
# ============================================================

# --- Shape and gravity field ---
SHAPE_OBJ = HERE / "data" / "shape_models" / "Bennu2019.obj"  # OSIRIS-REx shape (frame of the field)
SHAPE_UNIT = 1.0e3  # OBJ length unit -> m (the OBJ is in km)
BODY = "Bennu"
N_DEG = 4  # maximum SH degree fitted
G_CONST = 6.67430e-11  # [m^3 / (kg s^2)]

# --- Density groups ---
MASCON_MODE = "regions"  # "regions" (exact, 7 regions), "strategic" or "layered" (mascons)
CORE_RADIUS = 170.0  # regions: radius of the central ball [m] (~35 % of the volume)
BELT_LAT_DEG = 20.0  # regions: half-width in latitude of the equatorial belt [deg]
K_STRATEGIC = 7  # strategic: 7 = centre + +/-long + +/-intermediate + +/-pole
FRACTIONS = (0.65, 0.45, 0.45)  # strategic offsets as fractions of the PCA half-lengths
N_MASCONS = 12_000  # layered: mascons uniform in volume
N_LAYERS = 10  # layered: radial shells (one density each)
SEED_MASCONS = 1

# --- Measurement noise ---
SIGMA_COM = 1.0  # centre-of-mass offset per axis [m] -> degree-1 coefficients
SIGMA_CS = {2: 2.0e-5, 3: 5.0e-5, 4: 1.0e-4}  # normalized C_nm, S_nm; one entry per degree 2..N_DEG
SIGMA_MODEL = 1.6e-4  # representation error of the density model (n >= 2, in quadrature)
USE_GM = True  # GM observable (fixes the overall density scale)
SIGMA_GM = 6.0e-3  # GM [m^3/s^2]

# --- Prior: uniform box ---
RHO_LO = 500.0  # [kg/m^3]
RHO_HI = 2000.0  # [kg/m^3]

# --- MCMC ---
N_WALKERS = 48
N_STEPS = 10_000
BURN_IN = None  # None: 5 x max autocorrelation time
THIN = None  # None: max(1, min autocorrelation time / 2)
N_PROC = 1  # > 1: multiprocessing pool (the likelihood is ~10 us, so 1 is usually best)
N_STARTS = 8  # multi-start least squares from prior draws -> walker initialization
LOAD_CHAIN = None  # e.g. "results/get_interior_density_<stamp>/chain.npz" (replot only)

N_PRIOR = 20_000  # prior samples for the prior overlays
DISPLAY_SCALE = 1.0e-3  # kg/m^3 -> g/cm^3 in the figures

REGION_NAMES = ["core", "belt0-90E", "belt90-180E", "belt180-270E", "belt270-360E",
                "north", "south"]
REGION_SYM = [r"\mathrm{c}", r"\mathrm{e}_1", r"\mathrm{e}_2", r"\mathrm{e}_3", r"\mathrm{e}_4",
              r"\mathrm{N}", r"\mathrm{S}"]
STRATEGIC_NAMES = ["centre", "+long", "-long", "+mid", "-mid", "+pole", "-pole"]
STRATEGIC_SYM = [r"\mathrm{c}", r"+a", r"-a", r"+b", r"-b", r"+c", r"-c"]


# ============================================================
# Mesh volume
# ============================================================


def signed_volume_of_mesh(V: np.ndarray, T: np.ndarray) -> float:
    """
    Signed volume of a closed triangle mesh (sum of origin-apex tetrahedra).

    Parameters
    ----------
    V : ndarray, shape (n_v, 3)
    T : ndarray of int, shape (n_t, 3)

    Returns
    -------
    float
        Volume, positive for outward-oriented triangles.
    """
    v0 = V[T[:, 0]]
    v1 = V[T[:, 1]]
    v2 = V[T[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0)


# ============================================================
# Mascon model
# ============================================================


@dataclass
class Mascon:
    """
    Point mass of fixed position and volume.

    Attributes
    ----------
    pos : ndarray, shape (3,)
        Position in the shape frame [m].
    volume : float
        Volume represented by the mascon [m^3].
    layer : int
        Density group (index into theta).
    mass : float
        Mass [kg] (density * volume; only used by mascons_to_cs).
    """

    pos: np.ndarray  # (3,)
    volume: float  # [m^3]
    layer: int  # layer id
    mass: float = 0.0  # [kg] (set from density * volume)


# ============================================================
# Point-in-mesh (robust, fast if trimesh is installed)
# ============================================================


def build_trimesh(V: np.ndarray, T: np.ndarray):
    """
    trimesh.Trimesh of the shape (no processing), or None without trimesh.

    Parameters
    ----------
    V : ndarray, shape (n_v, 3)
    T : ndarray of int, shape (n_t, 3)

    Returns
    -------
    trimesh.Trimesh or None
    """
    if not _TRIMESH_AVAILABLE:
        return None
    return trimesh.Trimesh(vertices=V, faces=T, process=False)


# ---- fallback ray casting (dependency-free, slower) ----


def _tri_ray_intersect_moller(
    tri: np.ndarray, ray_o: np.ndarray, ray_d: np.ndarray, eps=1e-12
) -> bool:
    """
    Moeller-Trumbore test: does the ray o + t d (t > eps) hit the triangle?

    Parameters
    ----------
    tri : ndarray, shape (3, 3)
        Triangle vertices.
    ray_o, ray_d : ndarray, shape (3,)
        Ray origin and direction.
    eps : float
        Parallel-ray and minimum-distance tolerance.

    Returns
    -------
    bool
    """
    v0, v1, v2 = tri
    e1 = v1 - v0
    e2 = v2 - v0
    pvec = np.cross(ray_d, e2)
    det = float(np.dot(e1, pvec))
    if abs(det) < eps:
        return False
    inv_det = 1.0 / det
    tvec = ray_o - v0
    u = float(np.dot(tvec, pvec)) * inv_det
    if u < 0.0 or u > 1.0:
        return False
    qvec = np.cross(tvec, e1)
    v = float(np.dot(ray_d, qvec)) * inv_det
    if v < 0.0 or (u + v) > 1.0:
        return False
    t = float(np.dot(e2, qvec)) * inv_det
    return t > eps


def contains_points_ray_cast(V: np.ndarray, T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Parity test (odd number of hits) along a fixed non-axis ray direction.

    Works for closed watertight meshes; O(n_points x n_triangles) in pure Python, so much
    slower than trimesh.contains.

    Parameters
    ----------
    V : ndarray, shape (n_v, 3)
    T : ndarray of int, shape (n_t, 3)
    P : ndarray, shape (n, 3)
        Query points.

    Returns
    -------
    ndarray of bool, shape (n,)
        True inside.
    """
    ray_d = np.array([1.0, 0.137, 0.291], dtype=float)
    ray_d /= np.linalg.norm(ray_d)

    tris = V[T]  # (Nt,3,3)
    inside = np.zeros((P.shape[0],), dtype=bool)
    for i, p in enumerate(P):
        cnt = 0
        for tri in tris:
            if _tri_ray_intersect_moller(tri, p, ray_d):
                cnt += 1
        inside[i] = (cnt % 2) == 1
    return inside


def mesh_contains_points(
    mesh_tm, V: np.ndarray, T: np.ndarray, P: np.ndarray
) -> np.ndarray:
    """
    Inside test with trimesh when available, ray casting otherwise.

    Parameters
    ----------
    mesh_tm : trimesh.Trimesh or None
        From build_trimesh.
    V : ndarray, shape (n_v, 3)
    T : ndarray of int, shape (n_t, 3)
    P : ndarray, shape (n, 3)
        Query points.

    Returns
    -------
    ndarray of bool, shape (n,)
    """
    if mesh_tm is not None:
        # trimesh expects shape (N,3)
        return mesh_tm.contains(P)
    # fallback
    return contains_points_ray_cast(V, T, P)


# ============================================================
# Layered mascons: uniform in volume inside the mesh
# ============================================================


def sample_points_in_aabb(
    rng: np.random.Generator, bmin: np.ndarray, bmax: np.ndarray, n: int
) -> np.ndarray:
    """
    Uniform points in an axis-aligned box.

    Parameters
    ----------
    rng : numpy.random.Generator
    bmin, bmax : ndarray, shape (3,)
        Box corners.
    n : int
        Number of points.

    Returns
    -------
    ndarray, shape (n, 3)
    """
    return rng.uniform(bmin, bmax, size=(n, 3))


def sample_mascons_inside_mesh_fast(
    V: np.ndarray,
    T: np.ndarray,
    n_total: int,
    n_layers: int,
    rho_init: np.ndarray,
    layer_rule: str = "radial_bins",
    seed: int = 0,
    chunk: int = 5000,
    max_tries: int = 50,
):
    """
    Mascons uniform in volume inside a watertight mesh (trimesh), grouped in layers.

    trimesh.sample.volume_mesh can return fewer points than requested, so it is called
    repeatedly until n_total points are collected. Every mascon gets the volume
    V_mesh / n_total.

    Parameters
    ----------
    V : ndarray, shape (n_v, 3)
        Vertices [m].
    T : ndarray of int, shape (n_t, 3)
    n_total : int
        Number of mascons.
    n_layers : int
        Number of density groups (radial_bins).
    rho_init : array_like
        Initial density of each group [kg/m^3] (padded / truncated to n_layers); sets the
        mascon masses.
    layer_rule : {"radial_bins", "none"}
        "radial_bins": shells of equal thickness in |r| from the origin out to the largest
        vertex radius; "none": a single group.
    seed : int
        Seed of the (global numpy) RNG used by trimesh.
    chunk : int
        Points requested per volume_mesh call.
    max_tries : int
        Maximum number of volume_mesh calls.

    Returns
    -------
    list of Mascon

    Raises
    ------
    ValueError
        If the mesh is not watertight or has an invalid volume.
    RuntimeError
        If fewer than n_total points could be sampled.
    """
    rng = np.random.default_rng(seed)

    mesh = trimesh.Trimesh(vertices=V, faces=T, process=False)

    # Lightweight cleanup that sometimes fixes "almost watertight" meshes
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()

    # Try to fill holes (won't always fix)
    try:
        mesh.fill_holes()
    except Exception:
        pass

    if not mesh.is_watertight:
        raise ValueError(
            "Mesh is not watertight; trimesh.sample.volume_mesh may return too few points.\n"
            "Fix the OBJ (closed surface) or use the voxel fallback sampler."
        )

    # --- sample until we have n_total points ---
    np.random.seed(seed)  # trimesh uses numpy RNG internally
    pts_list = []
    got = 0
    tries = 0

    while got < n_total and tries < max_tries:
        tries += 1
        n_req = min(chunk, n_total - got)
        P = trimesh.sample.volume_mesh(mesh, n_req)

        # P can (rarely) come back shorter than requested; keep it anyway
        if P is None:
            P = np.zeros((0, 3), dtype=float)
        if P.ndim != 2 or P.shape[1] != 3:
            raise RuntimeError(
                f"Unexpected volume_mesh output shape: {getattr(P,'shape',None)}"
            )

        if P.shape[0] > 0:
            pts_list.append(P)
            got += P.shape[0]

        print(f"volume_mesh sampling: {got}/{n_total}", end="\r")

    print()

    if got < n_total:
        raise RuntimeError(
            f"volume_mesh could only generate {got} points after {tries} tries (wanted {n_total}).\n"
            "This usually indicates a mesh issue even if is_watertight=True, or numerical problems.\n"
            "Try voxel fallback."
        )

    P = np.vstack(pts_list)[:n_total]  # EXACT length

    # --- per-mascon volume ---
    Vmesh = float(mesh.volume)
    if not np.isfinite(Vmesh) or Vmesh <= 0.0:
        raise ValueError(f"Mesh volume invalid: {Vmesh}")
    vol_i = Vmesh / float(n_total)

    # --- assign layers ---
    rho_init = np.array(rho_init, dtype=float).reshape(-1)

    if layer_rule == "none":
        n_layers = 1
        layers = np.zeros((n_total,), dtype=int)
        if rho_init.size == 0:
            rho_init = np.array([1500.0])
        else:
            rho_init = rho_init[:1]

    elif layer_rule == "radial_bins":
        if n_layers < 1:
            n_layers = 1

        if rho_init.size != n_layers:
            if rho_init.size < n_layers:
                rho_init = np.pad(rho_init, (0, n_layers - rho_init.size), mode="edge")
            else:
                rho_init = rho_init[:n_layers]

        r = np.linalg.norm(P, axis=1)
        rmax = float(np.max(np.linalg.norm(V, axis=1)))  # safe upper bound
        edges = np.linspace(0.0, rmax, n_layers + 1)
        layers = np.clip(np.digitize(r, edges) - 1, 0, n_layers - 1).astype(int)

    else:
        raise ValueError("layer_rule must be 'none' or 'radial_bins'")

    # --- build mascons ---
    mascons = []
    for i in range(
        P.shape[0]
    ):  # NOTE: use len(P) not n_total (but len(P)==n_total anyway)
        lay = int(layers[i])
        mass = float(rho_init[lay]) * vol_i
        mascons.append(Mascon(pos=P[i], volume=vol_i, layer=lay, mass=mass))

    return mascons


# ============================================================
# Strategic big mascons on the principal axes
# ============================================================


def pca_body_axes(V: np.ndarray):
    """
    Principal axes of the vertex cloud.

    Parameters
    ----------
    V : ndarray, shape (n_v, 3)

    Returns
    -------
    ctr : ndarray, shape (3,)
        Vertex centroid.
    axes : ndarray, shape (3, 3)
        Columns = principal axes, by decreasing variance (column 0 = long axis).
    half : ndarray, shape (3,)
        Half-extents of the vertices along each axis.
    """
    ctr = np.mean(V, axis=0)
    X = V - ctr
    C = (X.T @ X) / max(1, X.shape[0] - 1)
    evals, evecs = np.linalg.eigh(C)  # ascending
    # sort descending variance -> axis 0 is "long" axis
    idx = np.argsort(evals)[::-1]
    axes = evecs[:, idx]

    # extents along axes
    proj = X @ axes  # (N,3)
    mins = np.min(proj, axis=0)
    maxs = np.max(proj, axis=0)
    half = 0.5 * (maxs - mins)
    return ctr, axes, half


def pull_point_inside(
    p: np.ndarray,
    ctr: np.ndarray,
    mesh_tm,
    V: np.ndarray,
    T: np.ndarray,
    max_iter: int = 60,
) -> np.ndarray:
    """
    Move a point toward ctr until it is inside the mesh (bisection on the scale).

    Parameters
    ----------
    p : ndarray, shape (3,)
        Candidate point.
    ctr : ndarray, shape (3,)
        Interior point (the centroid is inside for Bennu-like shapes).
    mesh_tm : trimesh.Trimesh or None
    V : ndarray, shape (n_v, 3)
    T : ndarray of int, shape (n_t, 3)
    max_iter : int
        Bisection steps.

    Returns
    -------
    ndarray, shape (3,)
        p if inside, otherwise ctr + alpha (p - ctr) with the largest inside alpha found.
    """
    p = np.array(p, dtype=float)
    ctr = np.array(ctr, dtype=float)

    # quick accept
    if bool(mesh_contains_points(mesh_tm, V, T, p.reshape(1, 3))[0]):
        return p

    # bisection on alpha in [0,1]: ctr + alpha*(p-ctr)
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        pmid = ctr + mid * (p - ctr)
        inside = bool(mesh_contains_points(mesh_tm, V, T, pmid.reshape(1, 3))[0])
        if inside:
            lo = mid  # can go further out
        else:
            hi = mid  # too far, pull in
    return ctr + lo * (p - ctr)


def build_strategic_big_mascons(
    V: np.ndarray,
    T: np.ndarray,
    K: int = 7,
    seed: int = 0,
    fractions: Tuple[float, float, float] = (0.65, 0.45, 0.45),
):
    """
    K big mascons of equal volume at strategic positions on the PCA body axes.

    Candidates, in this order: centre, +/- long-axis lobes, +/- intermediate axis,
    +/- short axis (poles), each pulled inside the mesh if needed. K < 7 truncates the list;
    K > 7 adds random interior points. Every mascon is its own density group.

    Parameters
    ----------
    V : ndarray, shape (n_v, 3)
        Vertices [m].
    T : ndarray of int, shape (n_t, 3)
    K : int
        Number of mascons.
    seed : int
        Seed of the extra interior points.
    fractions : tuple of float
        Offsets along the long, intermediate and short axes, as fractions of the vertex
        half-extents.

    Returns
    -------
    mascons : list of Mascon
        layer = mascon index, volume = V_mesh / K.
    rho_init : ndarray, shape (K,)
        Initial densities (1400 kg/m^3).
    """
    rng = np.random.default_rng(seed)
    mesh_tm = build_trimesh(V, T)

    ctr, axes, half = pca_body_axes(V)

    # build candidate positions in body principal-axis frame
    f1, f2, f3 = fractions
    a1 = half[0] * f1
    a2 = half[1] * f2
    a3 = half[2] * f3

    # helper: ctr + axes @ [dx,dy,dz]
    def p_body(dx, dy, dz):
        return ctr + axes @ np.array([dx, dy, dz], dtype=float)

    candidates = [
        p_body(0, 0, 0),  # core
        p_body(+a1, 0, 0),  # +long lobe
        p_body(-a1, 0, 0),  # -long lobe
        p_body(0, +a2, 0),  # +mid
        p_body(0, -a2, 0),  # -mid
        p_body(0, 0, +a3),  # +pole
        p_body(0, 0, -a3),  # -pole
    ]

    # ensure inside (shrink toward centroid if needed)
    cand_in = []
    for p in candidates:
        cand_in.append(pull_point_inside(p, ctr, mesh_tm, V, T))

    # truncate to K
    P = cand_in[:K]

    # if user asks for more than 7, add random interior points (still "big" but extra)
    if K > len(P):
        bmin = np.min(V, axis=0)
        bmax = np.max(V, axis=0)
        need = K - len(P)
        added = 0
        tries = 0
        while added < need and tries < 200:
            tries += 1
            Q = rng.uniform(bmin, bmax, size=(max(1000, 50 * need), 3))
            inside = mesh_contains_points(mesh_tm, V, T, Q)
            Q_in = Q[inside]
            if Q_in.shape[0] > 0:
                take = min(Q_in.shape[0], need - added)
                for j in range(take):
                    P.append(Q_in[j])
                added += take

        if added < need:
            raise RuntimeError(f"Could not find enough interior points for K={K}")

    P = np.asarray(P, dtype=float)

    # equal mascon volume = mesh volume / K
    if _TRIMESH_AVAILABLE:
        mesh = trimesh.Trimesh(vertices=V, faces=T, process=False)
        Vmesh = float(mesh.volume)
    else:
        Vmesh = abs(signed_volume_of_mesh(V, T))
    vol_i = Vmesh / float(K)

    # each mascon is its own "layer", so one rho per mascon is estimated
    mascons = []
    for i in range(K):
        mascons.append(Mascon(pos=P[i], volume=vol_i, layer=i, mass=0.0))

    rho_init = np.full((K,), 1400.0, dtype=float)

    return mascons, rho_init


# ============================================================
# Fully normalized Legendre + mascons -> Cbar / Sbar
# ============================================================


def fully_normalized_legendre(nmax: int, x: float) -> np.ndarray:
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


def mascons_to_cs(
    mascons: Iterable[Mascon],
    n_degree: int,
    ref_radius: float,
    exclude_c00: bool = True,
):
    """
    Fully normalized SH coefficients of a set of mascons (using their .mass).

    Cbar_nm + i Sbar_nm = sum_i (m_i / M) (r_i / R)^n Pbar_nm(sin phi_i) e^{i m lam_i} / (2n + 1).

    Parameters
    ----------
    mascons : iterable of Mascon
        Positions in the units of ref_radius.
    n_degree : int
        Maximum degree.
    ref_radius : float
        Reference radius R.
    exclude_c00 : bool
        Set C_00 to 0 (True) or 1 (False).

    Returns
    -------
    C, S : ndarray, shape (n_degree + 1, n_degree + 1)

    Raises
    ------
    ValueError
        If the total mass is not positive.
    """
    mascons = list(mascons)
    Mtot = float(sum(mc.mass for mc in mascons))
    if Mtot <= 0.0:
        raise ValueError("Total mass must be > 0")

    C = np.zeros((n_degree + 1, n_degree + 1), dtype=float)
    S = np.zeros((n_degree + 1, n_degree + 1), dtype=float)

    for mc in mascons:
        x, y, z = float(mc.pos[0]), float(mc.pos[1]), float(mc.pos[2])
        r = math.sqrt(x * x + y * y + z * z)
        if r == 0.0:
            continue
        lam = math.atan2(y, x)
        sinphi = z / r
        Pbar = fully_normalized_legendre(n_degree, sinphi)

        rho = r / ref_radius
        rpow = 1.0
        w = float(mc.mass) / Mtot

        for n in range(0, n_degree + 1):
            if n > 0:
                rpow *= rho
            for m in range(0, n + 1):
                if exclude_c00 and (n == 0 and m == 0):
                    continue
                # Cbar_nm = sum (m_i/M) (r_i/R)^n Pbar_nm [cos, sin](m lam_i) / (2n + 1)
                basis = rpow * Pbar[n, m] / (2 * n + 1)
                C[n, m] += w * basis * math.cos(m * lam)
                S[n, m] += w * basis * math.sin(m * lam)

    if exclude_c00:
        C[0, 0] = 0.0
        S[0, 0] = 0.0
    else:
        C[0, 0] = 1.0
        S[0, 0] = 0.0

    return C, S


def flatten_cs(
    C: np.ndarray, S: np.ndarray, n_degree: int, exclude_c00: bool = True
) -> np.ndarray:
    """
    Stack the coefficients as [C_nm ..., S_nm ...] in (n, m) order, n = 0..n_degree, m = 0..n.

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
    c_list, s_list = [], []
    for n in range(0, n_degree + 1):
        for m in range(0, n + 1):
            if exclude_c00 and (n == 0 and m == 0):
                continue
            c_list.append(C[n, m])
            s_list.append(S[n, m])
    return np.array(c_list + s_list, dtype=float)


def cs_names(n_degree: int, exclude_c00: bool = True) -> List[str]:
    """
    Labels of the entries of flatten_cs, e.g. 'C20', 'S22'.

    Parameters
    ----------
    n_degree : int
    exclude_c00 : bool

    Returns
    -------
    list of str
    """
    nm = [(n, m) for n in range(n_degree + 1) for m in range(n + 1)
          if not (exclude_c00 and n == 0 and m == 0)]
    return [f"C{n}{m}" for n, m in nm] + [f"S{n}{m}" for n, m in nm]


def build_mascon_design_matrix(
    mascons: List[Mascon], n_degree: int, ref_radius: float, exclude_c00: bool = True
) -> np.ndarray:
    """
    Design matrix of the SH coefficients in the mascon masses.

    flatten_cs(*mascons_to_cs(mascons)) == A @ m / sum(m), m = mascon masses.

    Parameters
    ----------
    mascons : list of Mascon
    n_degree : int
    ref_radius : float
    exclude_c00 : bool

    Returns
    -------
    ndarray, shape (2 n_terms, n_mascons)
    """
    mascons = list(mascons)
    Nmc = len(mascons)

    n_terms = 0
    for n in range(0, n_degree + 1):
        for m in range(0, n + 1):
            if exclude_c00 and (n == 0 and m == 0):
                continue
            n_terms += 1

    A = np.zeros((2 * n_terms, Nmc), dtype=float)

    for i, mc in enumerate(mascons):
        x, y, z = float(mc.pos[0]), float(mc.pos[1]), float(mc.pos[2])
        r = math.sqrt(x * x + y * y + z * z)
        if r == 0.0:
            continue
        lam = math.atan2(y, x)
        sinphi = z / r
        Pbar = fully_normalized_legendre(n_degree, sinphi)

        rho = r / ref_radius
        rpow = 1.0

        row_c = 0
        row_s = n_terms
        for n in range(0, n_degree + 1):
            if n > 0:
                rpow *= rho
            for m in range(0, n + 1):
                if exclude_c00 and (n == 0 and m == 0):
                    continue
                # Cbar_nm = sum (m_i/M) (r_i/R)^n Pbar_nm [cos, sin](m lam_i) / (2n + 1)
                basis = rpow * Pbar[n, m] / (2 * n + 1)
                A[row_c, i] = basis * math.cos(m * lam)
                A[row_s, i] = basis * math.sin(m * lam)
                row_c += 1
                row_s += 1

    return A


# ============================================================
# Density regions (exact quadrature)
# ============================================================


def region_labels(X, r_core, belt_lat_deg):
    """
    Region of each point: 0 core ball, 1-4 equatorial belt by longitude quadrant
    (0-90, 90-180, 180-270, 270-360 deg East), 5 north cap, 6 south cap.

    Parameters
    ----------
    X : ndarray, shape (Q, 3)
        Points [m] (body frame of the field).
    r_core : float
        Radius of the core ball [m] (0: no core).
    belt_lat_deg : float
        Half-width in latitude of the belt [deg].

    Returns
    -------
    ndarray of int, shape (Q,)
    """
    X = np.asarray(X, dtype=float)
    r = np.linalg.norm(X, axis=1)
    sin_lat = X[:, 2] / np.maximum(r, 1e-300)
    quad = np.floor((np.degrees(np.arctan2(X[:, 1], X[:, 0])) % 360.0) / 90.0).astype(int)
    lab = np.where(np.abs(sin_lat) < math.sin(math.radians(belt_lat_deg)), 1 + np.minimum(quad, 3),
                   np.where(sin_lat >= 0.0, 5, 6))
    return np.where(r < r_core, 0, lab)


def region_design_matrix(V, T, n_deg, r_ref, r_core, belt_lat_deg, chunk=8000):
    """
    Design matrix of the SH coefficients in the region masses, by exact quadrature.

    The polyhedron is the signed union of the origin-apex tetrahedra (0, v0, v1, v2) of its
    faces; a tetrahedron is mapped from the unit cube by x = t p(s, q),
    p = (1 - s) v0 + s (1 - q) v1 + s q v2, dV = 6 vol t^2 s dt ds dq. Each ray p(s, q)
    is split at t* = min(1, r_core / |p|): [0, t*] is in the core, [t*, 1] in the region of
    the direction p. Gauss-Legendre rules in t, s, q integrate the solid harmonics of degree
    <= n_deg exactly on each segment, so the only approximation is the region of each ray
    (to the face size on the latitude / longitude boundaries).

    Parameters
    ----------
    V : ndarray, shape (n_v, 3)
        Vertices [m].
    T : ndarray of int, shape (n_t, 3)
        Outward-oriented triangles.
    n_deg : int
        Maximum degree.
    r_ref : float
        Reference radius [m].
    r_core, belt_lat_deg : float
        Regions (region_labels).
    chunk : int
        Faces per batch (memory).

    Returns
    -------
    A : ndarray, shape (n_cs, 7)
        flatten_cs order (C_00 excluded): y = A w / sum(w), w = densities x volumes.
    volumes : ndarray, shape (7,)
        Region volumes [m^3].
    """
    nm = [(n, m) for n in range(n_deg + 1) for m in range(n + 1) if (n, m) != (0, 0)]
    K = 7
    A = np.zeros((2 * len(nm), K))
    volumes = np.zeros(K)
    s, ws = gauss01((n_deg + 3) // 2)
    q, wq = gauss01((n_deg + 2) // 2)
    u, wu = gauss01((n_deg + 4) // 2)
    SS, QQ = (g.ravel() for g in np.meshgrid(s, q, indexing="ij"))
    W_sq = 6.0 * np.outer(ws * s, wq).ravel()
    c0, c1, c2 = 1.0 - SS, SS * (1.0 - QQ), SS * QQ
    for i in range(0, len(T), chunk):
        v0, v1, v2 = (V[T[i : i + chunk, k]] for k in range(3))
        vol = np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0
        P = (c0[None, :, None] * v0[:, None] + c1[None, :, None] * v1[:, None]
             + c2[None, :, None] * v2[:, None]).reshape(-1, 3)
        w_p = (vol[:, None] * W_sq[None, :]).ravel()
        t_star = np.clip(r_core / np.linalg.norm(P, axis=1), 0.0, 1.0)
        lab_dir = region_labels(P, 0.0, belt_lat_deg)
        for a, b, lab in ((np.zeros_like(t_star), t_star, np.zeros_like(lab_dir)),
                          (t_star, np.ones_like(t_star), lab_dir)):
            L = b - a
            t = a[:, None] + L[:, None] * u[None, :]  # (n_p, n_u)
            w = ((w_p * L)[:, None] * wu[None, :] * t * t).ravel()
            X = (t[:, :, None] * P[:, None, :]).reshape(-1, 3)
            labs = np.repeat(lab, u.size)
            Vc, Vs = solid_harmonics(X, n_deg, r_ref)
            volumes += np.bincount(labs, w, K)
            for j, (n, m) in enumerate(nm):
                A[j] += np.bincount(labs, Vc[n, m] * w, K) / (2 * n + 1)
                A[j + len(nm)] += np.bincount(labs, Vs[n, m] * w, K) / (2 * n + 1)
    return A / volumes, volumes


def cs_sigma(n_deg, r_ref, sigma_com, sigma_cs, sigma_model):
    """
    Standard deviations of the flatten_cs entries (C_00 excluded).

    Degree 1: centre-of-mass offset sigma_com / (sqrt(3) R) (C_10 = z_cm / (sqrt(3) R), ...);
    degree n >= 2: hypot(sigma_cs[n], sigma_model).

    Parameters
    ----------
    n_deg : int
    r_ref : float
        Reference radius [m].
    sigma_com : float
        Centre-of-mass offset per axis [m].
    sigma_cs : dict
        Data sigma by degree (n >= 2).
    sigma_model : float
        Model representation error (n >= 2).

    Returns
    -------
    ndarray, shape (n_cs,)
    """
    deg = [n for n in range(n_deg + 1) for m in range(n + 1) if (n, m) != (0, 0)]
    sig = [sigma_com / (math.sqrt(3.0) * r_ref) if n == 1 else math.hypot(sigma_cs[n], sigma_model)
           for n in deg]
    return np.array(sig + sig, dtype=float)


# ============================================================
# Bennu gravity field (literature)
# ============================================================


def bennu_truth_cs_from_literature(ndeg: int):
    """
    Fully normalized gravity field of Bennu to degree 5 (zero above).

    Parameters
    ----------
    ndeg : int
        Size of the returned arrays (degree ndeg).

    Returns
    -------
    C, S : ndarray, shape (ndeg + 1, ndeg + 1)
        C_00 = 1.
    Rref : float
        Reference radius [m].
    GM : float
        [m^3/s^2].
    """
    Rref = 290.0
    GM = 4.890450

    C = np.zeros((ndeg + 1, ndeg + 1), dtype=float)
    S = np.zeros((ndeg + 1, ndeg + 1), dtype=float)
    C[0, 0] = 1.0

    J = {2: 1.926e-2, 3: -1.22e-3, 4: -6.50e-3, 5: 6.7e-5}
    for n, Jn in J.items():
        if n <= ndeg:
            C[n, 0] = -Jn

    if ndeg >= 2:
        C[2, 2] = 3.06e-3
        S[2, 2] = -1.09e-3

    if ndeg >= 3:
        C[3, 1] = 8.15e-4
        S[3, 1] = -5.43e-4
        C[3, 2] = -9.35e-4
        S[3, 2] = -5.38e-4
        C[3, 3] = 1.17e-3
        S[3, 3] = -3.1e-4

    if ndeg >= 4:
        C[4, 1] = -8.82e-4
        S[4, 1] = -5.8e-4
        C[4, 2] = -8.71e-4
        S[4, 2] = -8.4e-5
        C[4, 3] = -7.6e-5
        S[4, 3] = -3.9e-4
        C[4, 4] = 7.7e-4
        S[4, 4] = 2.25e-3

    if ndeg >= 5:
        C[5, 1] = -3.5e-4
        S[5, 1] = 1.6e-4
        C[5, 2] = -3.7e-5
        S[5, 2] = -2.7e-4
        C[5, 3] = -2.2e-6
        S[5, 3] = -9.5e-6
        C[5, 4] = 3.2e-4
        S[5, 4] = 5.0e-5
        C[5, 5] = -2.3e-5
        S[5, 5] = 3.0e-4

    return C, S, Rref, GM


# ============================================================
# Posterior (picklable)
# ============================================================


class DensityPosterior:
    """
    Log posterior of the group densities: Gaussian SH (+ GM) likelihood, uniform box prior.

    A top-level class with __call__ (not a closure), so that it pickles for a
    multiprocessing pool.

    Parameters
    ----------
    A : ndarray, shape (n_cs, n_mascons)
        build_mascon_design_matrix.
    volumes : ndarray, shape (n_mascons,)
        Mascon volumes [m^3].
    layers : ndarray of int, shape (n_mascons,)
        Density group of each mascon.
    y_obs : ndarray, shape (n_cs,)
        Observed coefficients (flatten_cs order).
    sigma_cs : float or ndarray, shape (n_cs,)
        Their standard deviations.
    lo, hi : ndarray, shape (K,)
        Prior bounds [kg/m^3].
    gm_obs : float, optional
        Observed GM [m^3/s^2]; None leaves GM out of the likelihood.
    sigma_gm : float
        Its standard deviation.
    """

    def __init__(self, A, volumes, layers, y_obs, sigma_cs, lo, hi, gm_obs=None, sigma_gm=1.0):
        self.A = np.asarray(A, dtype=float)
        self.volumes = np.asarray(volumes, dtype=float)
        self.layers = np.asarray(layers, dtype=int)
        self.y_obs = np.asarray(y_obs, dtype=float)
        if np.isscalar(sigma_cs):
            self.sigma = np.full_like(self.y_obs, float(sigma_cs))
        else:
            self.sigma = np.array(sigma_cs, dtype=float).reshape(-1)
            if self.sigma.shape[0] != self.y_obs.shape[0]:
                raise ValueError("sigma_cs has wrong length.")
        self.inv_var = 1.0 / (self.sigma * self.sigma)
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)
        self.gm_obs = gm_obs
        self.sigma_gm = float(sigma_gm)

    @property
    def ndim(self):
        """Number of density groups K."""
        return self.lo.size

    def mass(self, rho):
        """Total mass [kg] for group densities rho [kg/m^3]."""
        return float(np.sum(np.asarray(rho)[self.layers] * self.volumes))

    def y_model(self, rho):
        """
        Model SH coefficients (flatten_cs order); NaN if the total mass is not positive.

        Parameters
        ----------
        rho : ndarray, shape (K,)

        Returns
        -------
        ndarray, shape (n_cs,)
        """
        w = rho[self.layers] * self.volumes
        Mtot = float(np.sum(w))
        if Mtot <= 0.0:
            return np.full_like(self.y_obs, np.nan)
        return (self.A @ w) / Mtot

    def residuals(self, rho):
        """
        Normalized residuals (observed - model) / sigma: SH coefficients, then GM if used.

        Parameters
        ----------
        rho : ndarray, shape (K,)

        Returns
        -------
        ndarray
        """
        r = (self.y_obs - self.y_model(rho)) / self.sigma
        if self.gm_obs is not None:
            r = np.append(r, (self.gm_obs - G_CONST * self.mass(rho)) / self.sigma_gm)
        return r

    def log_prior(self, rho):
        """0 inside the box [lo, hi], -inf outside."""
        if np.any(rho < self.lo) or np.any(rho > self.hi):
            return -np.inf
        return 0.0

    def log_like(self, rho):
        """Gaussian log likelihood (up to a constant); -inf for a non-finite model."""
        ym = self.y_model(rho)
        if not np.all(np.isfinite(ym)):
            return -np.inf
        r = self.y_obs - ym
        ll = -0.5 * float(np.sum(r * r * self.inv_var))
        if self.gm_obs is not None:
            ll -= 0.5 * ((self.gm_obs - G_CONST * self.mass(rho)) / self.sigma_gm) ** 2
        return ll

    def __call__(self, rho):
        """Log posterior of rho [kg/m^3]."""
        rho = np.asarray(rho, dtype=float)
        lp = self.log_prior(rho)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_like(rho)


def initialize_walkers(post, n_walkers, n_starts, rng, max_draws=100_000):
    """
    Multi-start bounded least squares from prior draws, then walkers around the best fit.

    Walkers are drawn from N(rho_lsq, cov), cov = (J^T J)^-1 at the best fit (a small
    diagonal if J^T J is singular), and redrawn until they are inside the prior box.

    Parameters
    ----------
    post : DensityPosterior
    n_walkers : int
    n_starts : int
        Number of least-squares starts.
    rng : numpy.random.Generator
    max_draws : int
        Draws before falling back to a small ball around the best fit.

    Returns
    -------
    p0 : ndarray, shape (n_walkers, K)
    rho_lsq : ndarray, shape (K,)
        Best least-squares fit.
    cov_lsq : ndarray, shape (K, K) or None
        Its covariance (None if singular).
    """
    width = post.hi - post.lo
    best = None
    for k in range(n_starts):
        x0 = post.lo + rng.random(post.ndim) * width
        sol = least_squares(
            post.residuals, x0, bounds=(post.lo, post.hi), x_scale=width, method="trf",
            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400,
        )
        print(f"[Init] start {k + 1:2d}/{n_starts}: chi2 = {2 * sol.cost:14.4f}")
        if best is None or sol.cost < best.cost:
            best = sol
    try:
        cov = np.linalg.inv(best.jac.T @ best.jac)
        L = np.linalg.cholesky(0.5 * (cov + cov.T))
    except np.linalg.LinAlgError:
        cov, L = None, np.diag(1e-4 * width)
    p0, draws = [], 0
    while len(p0) < n_walkers:
        draws += 1
        if draws > max_draws:
            L = np.diag(1e-4 * width)
        th = best.x + L @ rng.standard_normal(post.ndim)
        if np.isfinite(post(th)):
            p0.append(th)
    print(f"[Init] best fit: chi2 = {2 * best.cost:.4f} ({best.fun.size} observables, "
          f"{post.ndim} parameters)")
    return np.array(p0), best.x, cov


# ============================================================
# Figures
# ============================================================


def plot_traces(chain, logp, burn, ref=None, name="fig_traces"):
    """
    Walker traces of every density and of log p (burn-in shaded).

    Parameters
    ----------
    chain : ndarray, shape (n_step, n_walk, K)
        Internal units (scaled by MCMCPlots.PARAM_SCALE for display).
    logp : ndarray, shape (n_step, n_walk)
    burn : int
        Burn-in steps (shaded).
    ref : ndarray, shape (K,), optional
        Reference values (dashed), e.g. the least-squares fit.
    name : str
        File name of the PDF.

    Returns
    -------
    Path or None
        The PDF written.
    """
    n_step, n_walk, n_par = chain.shape
    s = MCMCPlots.PARAM_SCALE
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
                ax.axhline(ref[i] * s[i], color=MCMCPlots.C_BATCH, ls="--", lw=0.8)
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


def plot_densities(post, rho_lsq, cov_lsq, rho_bulk, lo, hi, name="fig_densities"):
    """
    Density of every group: posterior median with 68 / 95 % intervals, least squares, bulk.

    Parameters
    ----------
    post : ndarray, shape (N, K)
        Posterior samples [kg/m^3].
    rho_lsq : ndarray, shape (K,)
        Least-squares fit.
    cov_lsq : ndarray, shape (K, K) or None
        Its covariance (1-sigma error bars when given).
    rho_bulk : float
        Bulk density of the homogeneous body (GM / (G V)).
    lo, hi : float
        Prior bounds (dashed).
    name : str

    Returns
    -------
    Path or None
    """
    s = MCMCPlots.PARAM_SCALE[0]
    K = post.shape[1]
    k = np.arange(K)
    q = np.percentile(post * s, [2.5, 16, 50, 84, 97.5], axis=0)
    fig, ax = plt.subplots(figsize=(MCMCPlots.COL_W if K <= 8 else MCMCPlots.TXT_W, 2.2),
                           layout="constrained")
    ax.axhspan(lo * s, hi * s, color=MCMCPlots.C_SPAN, lw=0, zorder=0)
    ax.axhline(rho_bulk * s, color=MCMCPlots.C_TRUTH, ls=":", lw=1.0, label="Bulk (Homogeneous)")
    ax.vlines(k - 0.12, q[0], q[4], color=MCMCPlots.C_POST, lw=0.8)
    ax.vlines(k - 0.12, q[1], q[3], color=MCMCPlots.C_POST, lw=2.4)
    ax.plot(k - 0.12, q[2], "o", ms=3.2, mfc="white", mec=MCMCPlots.C_POST, mew=0.9,
            label=r"MCMC Posterior (68, 95$\%$)")
    err = None
    if cov_lsq is not None:  # no bar where the 1-sigma exceeds the prior width (degenerate fit)
        err = np.sqrt(np.clip(np.diag(cov_lsq), 0.0, None)) * s
        err = np.where(err < (hi - lo) * s, err, np.nan)
    ax.errorbar(k + 0.12, rho_lsq * s, yerr=err, fmt="s", ms=2.8, color=MCMCPlots.C_BATCH,
                lw=0.9, capsize=0, label="Batch Least Squares")
    ax.set_xticks(k, [f"${MCMCPlots.PARAM_SYM[i]}$" for i in k])
    ax.set_xlim(-0.6, K - 0.4)
    pad = 0.08 * (q[4].max() - q[0].min() + 1e-12)
    ax.set_ylim(min(q[0].min(), rho_bulk * s, np.min(rho_lsq) * s) - pad,
                max(q[4].max(), rho_bulk * s, np.max(rho_lsq) * s) + pad)
    ax.set_ylabel(f"$\\rho$ [{MCMCPlots.PARAM_UNIT[0]}]")
    fig.legend(loc="outside upper center", ncol=3 if K > 8 else 2)
    return MCMCPlots.save_figure(fig, name)


def plot_density_sections(V, T, pos, layers, rho_med, com, name="fig_density_sections",
                          slab=0.15):
    """
    Posterior median density of the mascons on three cuts through the origin.

    With at most 50 mascons every mascon is drawn, projected on each plane and labelled
    with its group symbol; otherwise only the mascons within +/- slab R_max of the plane.

    Parameters
    ----------
    V : ndarray, shape (n_v, 3)
        Vertices [m].
    T : ndarray of int, shape (n_t, 3)
    pos : ndarray, shape (n_mascons, 3)
        Mascon positions [m].
    layers : ndarray of int, shape (n_mascons,)
    rho_med : ndarray, shape (K,)
        Posterior median densities [kg/m^3].
    com : ndarray, shape (3,)
        Centre of mass of the posterior median [m].
    name : str
    slab : float
        Half-thickness of the slab around each plane, as a fraction of the largest vertex
        radius (many-mascon case).

    Returns
    -------
    Path or None
    """
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    s = MCMCPlots.PARAM_SCALE[0]
    dens = rho_med[layers] * s
    few = len(pos) <= 50
    norm = plt.Normalize(dens.min(), dens.max() if dens.max() > dens.min() else dens.min() + 1e-9)
    cmap = plt.get_cmap("cividis")
    r_max = np.linalg.norm(V, axis=1).max()
    fig, axs = plt.subplots(1, 3, figsize=(MCMCPlots.TXT_W, 2.7), layout="constrained")
    axes_names = ["x", "y", "z"]
    for ax, cut in zip(axs, (2, 1, 0)):  # planes z = 0, y = 0, x = 0
        i, j = [k for k in range(3) if k != cut]
        ax.add_collection(LineCollection(mesh_section(V, T, cut), colors="0.35", lw=0.6))
        sel = np.ones(len(pos), bool) if few else np.abs(pos[:, cut]) < slab * r_max
        order = np.argsort(pos[sel, cut]) if few else slice(None)  # nearer mascons on top
        P, D = pos[sel][order], dens[sel][order]
        if few:
            ax.scatter(P[:, i], P[:, j], c=D, cmap=cmap, norm=norm, s=70, ec="k", lw=0.5, zorder=3)
            for p, lay in zip(P, layers[sel][order]):
                ax.annotate(f"${MCMCPlots.PARAM_SYM[lay]}$", (p[i], p[j]), xytext=(5, 4),
                            textcoords="offset points", fontsize=6, color="0.2", zorder=4)
        else:
            ax.scatter(P[:, i], P[:, j], c=D, cmap=cmap, norm=norm, s=3, lw=0, rasterized=True)
        ax.plot(com[i], com[j], "*", ms=7, mfc=MCMCPlots.C_TRUTH, mec="white", mew=0.4, zorder=5)
        ax.plot(0, 0, "+", ms=6, color="0.35", mew=0.6)
        ax.set_aspect("equal")
        lim = 1.05 * np.abs(V[:, [i, j]]).max()
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"${axes_names[i]}$ [m]")
        ax.set_ylabel(f"${axes_names[j]}$ [m]")
        ax.set_title(f"${axes_names[cut]} = 0$" + ("" if few else rf" ($\pm${slab * r_max:.0f} m)"))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, shrink=0.85, pad=0.01)
    cb.set_label(f"$\\rho$ [{MCMCPlots.PARAM_UNIT[0]}] (Posterior Median)")
    handles = [
        Line2D([], [], color="0.35", lw=0.6, label=f"{BODY} Surface"),
        Line2D([], [], ls="", marker="o", mfc=cmap(0.6), mec="k" if few else cmap(0.6), mew=0.5,
               ms=6 if few else 3, label="Mascons"),
        Line2D([], [], ls="", marker="*", mfc=MCMCPlots.C_TRUTH, mec="white", ms=7,
               label="Centre of Mass (Posterior Median)"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3)
    return MCMCPlots.save_figure(fig, name)


def _plane_section(V, T, e1, e2):
    """
    Cut of the surface mesh by the plane through the origin spanned by e1, e2.

    Parameters
    ----------
    V : ndarray, shape (n_v, 3)
    T : ndarray of int, shape (n_t, 3)
    e1, e2 : ndarray, shape (3,)
        Orthonormal in-plane axes.

    Returns
    -------
    ndarray, shape (n, 2, 2)
        Segments in the in-plane coordinates (u, v) = (x . e1, x . e2).
    """
    nrm = np.cross(e1, e2)
    R = np.array([e1, e2, nrm])  # rows: new axes; the plane is the new z = 0
    return mesh_section(V @ R.T, T, 2)


def _inside_mask(segs, u, v):
    """
    Grid points inside a closed planar section (even-odd rule, one scanline per row).

    Parameters
    ----------
    segs : ndarray, shape (n, 2, 2)
        Outline segments (_plane_section).
    u, v : ndarray, shape (n_u,), (n_v,)
        Grid coordinates.

    Returns
    -------
    ndarray of bool, shape (n_v, n_u)
    """
    a, b = segs[:, 0], segs[:, 1]
    mask = np.zeros((v.size, u.size), dtype=bool)
    for k, vk in enumerate(v):
        hit = (a[:, 1] > vk) != (b[:, 1] > vk)
        lam = (vk - a[hit, 1]) / (b[hit, 1] - a[hit, 1])
        uc = np.sort(a[hit, 0] + lam * (b[hit, 0] - a[hit, 0]))
        mask[k] = (np.searchsorted(uc, u) % 2) == 1
    return mask


def plot_region_sections(V, T, rho_med, com, r_core, belt_lat_deg, name="fig_density_sections",
                         h=2.0):
    """
    Posterior median density of the regions on three planes through the origin.

    Planes: equator z = 0 (core and the four belt quadrants), and the meridian planes
    through longitudes 45 / 225 deg and 135 / 315 deg (core, two belt quadrants, north and
    south caps). Region boundaries dashed, region symbols at their in-plane centroids.

    Parameters
    ----------
    V : ndarray, shape (n_v, 3)
        Vertices [m].
    T : ndarray of int, shape (n_t, 3)
    rho_med : ndarray, shape (7,)
        Posterior median densities [kg/m^3].
    com : ndarray, shape (3,)
        Centre of mass of the posterior median [m].
    r_core, belt_lat_deg : float
        Regions (region_labels).
    name : str
    h : float
        Grid step [m].

    Returns
    -------
    Path or None
    """
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    sc = MCMCPlots.PARAM_SCALE[0]
    dens = rho_med * sc
    norm = plt.Normalize(dens.min(), dens.max() if dens.max() > dens.min() else dens.min() + 1e-9)
    cmap = plt.get_cmap("cividis")
    lim = 1.05 * np.linalg.norm(V, axis=1).max()
    g = np.arange(-lim, lim + h, h)
    U, W = np.meshgrid(g, g)
    z_hat = np.array([0.0, 0.0, 1.0])
    planes = [(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), r"$z = 0$", "$x$ [m]", "$y$ [m]")]
    for lon in (45.0, 135.0):
        c, s_ = math.cos(math.radians(lon)), math.sin(math.radians(lon))
        planes.append((np.array([c, s_, 0.0]), z_hat,
                       rf"$\lambda = {lon:.0f}^\circ / {lon + 180:.0f}^\circ$",
                       rf"Along $\lambda = {lon:.0f}^\circ$ [m]", "$z$ [m]"))
    tb = math.tan(math.radians(belt_lat_deg))
    fig, axs = plt.subplots(1, 3, figsize=(MCMCPlots.TXT_W, 2.6), layout="constrained")
    for ip, (ax, (e1, e2, title, xl, yl)) in enumerate(zip(axs, planes)):
        segs = _plane_section(V, T, e1, e2)
        inside = _inside_mask(segs, g, g)
        X = U.ravel()[:, None] * e1 + W.ravel()[:, None] * e2
        lab = region_labels(X, r_core, belt_lat_deg).reshape(U.shape)
        img = np.ma.masked_where(~inside, dens[lab])
        ax.imshow(img, origin="lower", extent=(g[0] - h / 2, g[-1] + h / 2, g[0] - h / 2, g[-1] + h / 2),
                  cmap=cmap, norm=norm, interpolation="nearest", rasterized=True)
        ax.add_collection(LineCollection(segs, colors="0.2", lw=0.6))
        # region boundaries: core circle, then rays from the core to the surface
        th = np.linspace(0.0, 2.0 * np.pi, 361)
        ax.plot(r_core * np.cos(th), r_core * np.sin(th), color="w", ls="--", lw=0.6)
        dirs = ([(1, 0), (0, 1), (-1, 0), (0, -1)] if ip == 0 else
                [(sx, sy * tb) for sx in (1, -1) for sy in (1, -1)])
        for dx, dy in dirs:
            d = np.array([dx, dy]) / math.hypot(dx, dy)
            r = np.arange(r_core, lim, h / 2)
            ii = np.clip(np.round((r * d[1] - g[0]) / h).astype(int), 0, g.size - 1)
            jj = np.clip(np.round((r * d[0] - g[0]) / h).astype(int), 0, g.size - 1)
            out = np.flatnonzero(~inside[ii, jj])
            r_end = r[out[0]] if out.size else r[-1]
            ax.plot([r_core * d[0], r_end * d[0]], [r_core * d[1], r_end * d[1]],
                    color="w", ls="--", lw=0.6)
        for k in np.unique(lab[inside]):
            sel = inside & (lab == k)
            xy = (0.0, -0.55 * r_core) if k == 0 else (U[sel].mean(), W[sel].mean())
            ax.text(*xy, f"${MCMCPlots.PARAM_SYM[k]}$", ha="center", va="center", fontsize=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
        ax.plot(com @ e1, com @ e2, "*", ms=7, mfc=MCMCPlots.C_TRUTH, mec="white", mew=0.4, zorder=5)
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, shrink=0.85, pad=0.01)
    cb.set_label(f"$\\rho$ [{MCMCPlots.PARAM_UNIT[0]}] (Posterior Median)")
    handles = [
        Line2D([], [], color="0.2", lw=0.6, label=f"{BODY} Surface"),
        Line2D([], [], color="0.5", ls="--", lw=0.6, label="Region Boundaries"),
        Line2D([], [], ls="", marker="*", mfc=MCMCPlots.C_TRUTH, mec="white", ms=7,
               label="Centre of Mass (Posterior Median)"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3)
    return MCMCPlots.save_figure(fig, name)


def plot_residuals(post_model, post, rho_lsq, rho_hom, names, name="fig_residuals", n_draw=300):
    """
    Normalized residuals of every observable: posterior predictive band, least squares and
    homogeneous body (+/-3 sigma shaded; symlog scale beyond 10 sigma).

    Parameters
    ----------
    post_model : DensityPosterior
    post : ndarray, shape (N, K)
        Posterior samples.
    rho_lsq, rho_hom : ndarray, shape (K,)
        Least-squares fit and homogeneous body.
    names : list of str
        Observable names (DensityPosterior.residuals order).
    name : str
    n_draw : int
        Posterior draws for the predictive band.

    Returns
    -------
    Path or None
    """
    keep = np.array([not (nm.startswith("S") and nm.endswith("0") and len(nm) == 3)
                     for nm in names])  # S_n0 == 0 identically
    rng = np.random.default_rng(2)
    draws = post[rng.choice(len(post), size=min(n_draw, len(post)), replace=False)]
    R = np.array([post_model.residuals(th) for th in draws])[:, keep]
    r_lsq = post_model.residuals(rho_lsq)[keep]
    r_hom = post_model.residuals(rho_hom)[keep]
    names = [nm for nm, k_ in zip(names, keep) if k_]
    k = np.arange(len(names))
    lo, med, hi = np.percentile(R, [2.5, 50, 97.5], axis=0)
    fig, ax = plt.subplots(figsize=(MCMCPlots.TXT_W, 2.4), layout="constrained")
    ax.axhspan(-3, 3, color=MCMCPlots.C_SPAN, lw=0, zorder=0)
    ax.vlines(k, lo, hi, color=MCMCPlots.C_POST, lw=1.6, alpha=0.5)
    ax.plot(k, med, "o", ms=2.4, color=MCMCPlots.C_POST, label=r"MCMC Posterior (median, 95$\%$)")
    ax.plot(k, r_lsq, "+", ms=3.5, mew=0.8, color=MCMCPlots.C_SC, label="Batch Least Squares")
    ax.plot(k, r_hom, "x", ms=3.0, mew=0.8, color=MCMCPlots.C_BATCH, label="Equal Densities (Bulk)")
    if max(np.abs(lo).max(), np.abs(hi).max(), np.abs(r_hom).max()) > 10.0:
        ax.set_yscale("symlog", linthresh=3.0)
    fs = 6 if len(k) <= 60 else 4
    ax.set_xticks(k, names, rotation=90, fontsize=fs)
    ax.set_xlim(-0.7, k[-1] + 0.7)
    ax.set_ylabel(r"(Observed $-$ Model) / $\sigma$")
    fig.legend(loc="outside upper center", ncol=3)
    return MCMCPlots.save_figure(fig, name)


def write_summary(path, post, rho_lsq, cov_lsq, rho_hom, par_names, extra_lines):
    """
    Text table of the densities: posterior median and 16-84 % half-widths, least squares,
    homogeneous body. Printed and written to path.

    Parameters
    ----------
    path : Path
    post : ndarray, shape (N, K)
        Posterior samples [kg/m^3].
    rho_lsq : ndarray, shape (K,)
    cov_lsq : ndarray, shape (K, K) or None
    rho_hom : ndarray, shape (K,)
    par_names : list of str
    extra_lines : list of str
        Header lines.
    """
    sig = (np.full(len(rho_lsq), np.nan) if cov_lsq is None
           else np.sqrt(np.clip(np.diag(cov_lsq), 0.0, None)))
    lines = list(extra_lines) + ["", "Densities [kg/m^3] (posterior median, 16-84 % half-widths)",
                                 f"{'group':>12s} {'median':>12s} {'-1sig':>12s} {'+1sig':>12s} "
                                 f"{'LSQ':>12s} {'LSQ 1sig':>12s} {'homogeneous':>12s}"]
    for k, nm in enumerate(par_names):
        q16, q50, q84 = np.percentile(post[:, k], [16, 50, 84])
        lines.append(f"{nm:>12s} {fmt(q50)} {fmt(q50 - q16)} {fmt(q84 - q50)} "
                     f"{fmt(rho_lsq[k])} {fmt(sig[k])} {fmt(rho_hom[k])}")
    text = "\n".join(lines) + "\n"
    print(text)
    MCMCPlots.write_output(path, text.encode(), tag="Summary")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    t_start = time.time()

    # --- shape ---
    V, T = load_shape(SHAPE_OBJ, SHAPE_UNIT)
    vol = abs(signed_volume_of_mesh(V, T))
    print(f"[Shape] {SHAPE_OBJ.name}: {len(V)} vertices, {len(T)} faces, volume {vol:.6e} m^3")

    # --- observed field ---
    C_obs, S_obs, R_REF, GM_OBS = bennu_truth_cs_from_literature(N_DEG)
    y_obs = flatten_cs(C_obs, S_obs, N_DEG, exclude_c00=True)
    names = cs_names(N_DEG) + (["GM"] if USE_GM else [])
    print(f"[Data] {BODY} field: R_ref = {R_REF} m, GM = {GM_OBS} m^3/s^2, degree {N_DEG} "
          f"({y_obs.size} coefficients{', + GM' if USE_GM else ''})")

    # --- density groups (design matrix from the saved chain when replotting) ---
    saved = np.load(LOAD_CHAIN, allow_pickle=True) if LOAD_CHAIN else None
    t0 = time.time()
    A = None
    r_core, belt_lat = CORE_RADIUS, BELT_LAT_DEG
    if saved is not None:
        pos, volumes, layers = saved["pos"], saved["volumes"], saved["layers"].astype(int)
        mode = str(saved["mascon_mode"])
        if "A" in saved.files:
            A = saved["A"]
        if mode == "regions":
            r_core, belt_lat = float(saved["core_radius"]), float(saved["belt_lat_deg"])
    elif MASCON_MODE == "regions":
        mode = MASCON_MODE
        A, volumes = region_design_matrix(V, T, N_DEG, R_REF, r_core, belt_lat)
        layers, pos = np.arange(volumes.size), np.zeros((0, 3))
    else:
        mode = MASCON_MODE
        if mode == "strategic":
            mascons, _ = build_strategic_big_mascons(
                V, T, K=K_STRATEGIC, seed=SEED_MASCONS, fractions=FRACTIONS
            )
        elif mode == "layered":
            mascons = sample_mascons_inside_mesh_fast(
                V, T, n_total=N_MASCONS, n_layers=N_LAYERS, rho_init=np.full(N_LAYERS, 1400.0),
                layer_rule="radial_bins", seed=SEED_MASCONS,
            )
        else:
            raise ValueError("MASCON_MODE must be 'regions', 'strategic' or 'layered'")
        pos = np.array([mc.pos for mc in mascons], dtype=float)
        volumes = np.array([mc.volume for mc in mascons], dtype=float)
        layers = np.array([mc.layer for mc in mascons], dtype=int)
    K = int(layers.max()) + 1
    if mode == "regions":
        print(f"[Regions] core r < {r_core:g} m, belt |lat| < {belt_lat:g} deg: volume "
              f"fractions {np.array2string(volumes / volumes.sum(), precision=3)} "
              f"({time.time() - t0:.1f} s)")
    else:
        print(f"[Mascons] {mode}: {len(pos)} mascons, {K} density groups, "
              f"max |r| {np.linalg.norm(pos, axis=1).max():.1f} m (vertices "
              f"{np.linalg.norm(V, axis=1).max():.1f} m) ({time.time() - t0:.1f} s)")

    if mode == "regions":
        par_names, sym = REGION_NAMES, REGION_SYM
    elif mode == "strategic":
        par_names = (STRATEGIC_NAMES + [f"extra{k}" for k in range(7, K)])[:K]
        sym = (STRATEGIC_SYM + [str(k) for k in range(7, K)])[:K]
    else:
        par_names = [f"shell{k}" for k in range(K)]
        sym = [str(k) for k in range(K)]
    param_sym = [rf"\rho_{{{s_}}}" for s_ in sym]
    MCMCPlots.configure(param_sym, ["g/cm$^3$"] * K, np.full(K, DISPLAY_SCALE), body=BODY)

    # --- posterior ---
    if A is None:
        mascons = [Mascon(pos=p, volume=v, layer=int(l)) for p, v, l in zip(pos, volumes, layers)]
        A = build_mascon_design_matrix(mascons, N_DEG, R_REF, exclude_c00=True)
    sigma = cs_sigma(N_DEG, R_REF, SIGMA_COM, SIGMA_CS, SIGMA_MODEL)
    n_info = int(np.sum(np.abs(A).max(axis=1) > 0.0)) + int(USE_GM)  # S_n0 rows are zero
    dof = n_info - K
    lo, hi = np.full(K, RHO_LO), np.full(K, RHO_HI)
    model = DensityPosterior(A, volumes, layers, y_obs, sigma, lo, hi,
                             gm_obs=GM_OBS if USE_GM else None, sigma_gm=SIGMA_GM)
    rho_bulk = GM_OBS / (G_CONST * float(np.sum(volumes)))
    rho_hom = np.full(K, rho_bulk)
    r_hom = model.residuals(rho_hom)
    print(f"[Data] homogeneous body ({rho_bulk:.2f} kg/m^3): chi2 = {r_hom @ r_hom:.4g} "
          f"({n_info} informative observables, {dof} degrees of freedom)")
    t0 = time.time()
    for _ in range(500):
        model(rho_hom)
    print(f"[Model] log posterior: {(time.time() - t0) / 500 * 1e6:.1f} us per call")

    fig_dir = MCMCPlots.set_figure_dir(
        HERE / "results" / (Path(__file__).stem + "_" + datetime.now().strftime("%y%m%d%H%M%S"))
    )

    # --- sampling ---
    rng = np.random.default_rng()
    if saved is not None:
        chain, logp = saved["chain"].astype(float), saved["log_prob"].astype(float)
        tau, rho_lsq = saved["tau"], saved["rho_lsq"]
        cov_lsq = saved["cov_lsq"] if saved["cov_lsq"].ndim == 2 else None
        print(f"[MCMC] loaded {LOAD_CHAIN}: chain {chain.shape}")
    else:
        p0, rho_lsq, cov_lsq = initialize_walkers(model, N_WALKERS, N_STARTS, rng)
        moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
        if N_PROC > 1:
            from multiprocessing import Pool

            with Pool(N_PROC) as pool:
                sampler = emcee.EnsembleSampler(N_WALKERS, K, model, moves=moves, pool=pool)
                sampler.run_mcmc(p0, N_STEPS, progress=True)
        else:
            sampler = emcee.EnsembleSampler(N_WALKERS, K, model, moves=moves)
            sampler.run_mcmc(p0, N_STEPS, progress=True)
        chain, logp = sampler.get_chain(), sampler.get_log_prob()
        tau = sampler.get_autocorr_time(tol=0)
        print(f"[MCMC] acceptance {np.mean(sampler.acceptance_fraction):.3f}")
        buf = io.BytesIO()
        np.savez_compressed(
            buf, chain=chain, log_prob=logp, tau=tau, rho_lsq=rho_lsq,
            cov_lsq=np.array(np.nan) if cov_lsq is None else cov_lsq,
            pos=pos, volumes=volumes, layers=layers, mascon_mode=mode, A=A,
            core_radius=r_core, belt_lat_deg=belt_lat,
            par_names=np.array(par_names), y_obs=y_obs, gm_obs=GM_OBS, use_gm=USE_GM,
            sigma=sigma, sigma_com=SIGMA_COM, sigma_model=SIGMA_MODEL, sigma_gm=SIGMA_GM,
            n_deg=N_DEG, r_ref=R_REF,
            shape=str(SHAPE_OBJ.name), shape_unit=SHAPE_UNIT, rho_lo=RHO_LO, rho_hi=RHO_HI,
        )
        MCMCPlots.write_output(fig_dir / "chain.npz", buf.getvalue(), tag="Chain")

    burn = int(5 * np.max(tau)) if BURN_IN is None else int(BURN_IN)
    thin = max(1, int(0.5 * np.min(tau))) if THIN is None else int(THIN)
    if burn >= chain.shape[0]:
        print(f"[MCMC] WARNING: burn-in {burn} >= {chain.shape[0]} steps; using half the chain")
        burn = chain.shape[0] // 2
    post = chain[burn::thin].reshape(-1, K)
    print(f"[MCMC] tau = {np.array2string(tau, precision=1)}; burn-in {burn}, thin {thin}, "
          f"{len(post)} samples ({chain.shape[0] / np.max(tau):.0f} tau)")

    rho_med = np.median(post, axis=0)
    y_med = dict(zip(cs_names(N_DEG), model.y_model(rho_med)))  # degree 1 = centre of mass
    com = math.sqrt(3.0) * R_REF * np.array([y_med["C11"], y_med["S11"], y_med["C10"]])

    # --- figures ---
    prior = [uniform(RHO_LO, RHO_HI - RHO_LO) for _ in range(K)]
    # a least-squares covariance wider than the prior box (e.g. the density-scale ridge
    # without GM) would only rescale the corner axes: drop it from the corner
    cov_plot = cov_lsq
    if cov_lsq is not None and np.any(np.sqrt(np.abs(np.diag(cov_lsq))) > RHO_HI - RHO_LO):
        cov_plot = None
        print("[Plot] least-squares covariance wider than the prior box: not drawn")
    plot_traces(chain, logp, burn, ref=rho_lsq)
    MCMCPlots.plot_corner(post, batch_mean=rho_lsq, batch_cov=cov_plot, name="fig_corner")
    MCMCPlots.plot_corner(post, batch_mean=rho_lsq, batch_cov=cov_plot, prior=prior,
                          name="fig_corner_prior")
    plot_densities(post, rho_lsq, cov_lsq, rho_bulk, RHO_LO, RHO_HI)
    if mode == "regions":
        plot_region_sections(V, T, rho_med, com, r_core, belt_lat)
    else:
        plot_density_sections(V, T, pos, layers, rho_med, com)
    plot_residuals(model, post, rho_lsq, rho_hom, names)

    chi2_med = float(np.sum(model.residuals(rho_med) ** 2))
    write_summary(
        fig_dir / "summary.txt", post, rho_lsq, cov_lsq, rho_hom, par_names,
        [
            f"Shape {SHAPE_OBJ.name}: volume {vol:.6e} m^3; field R_ref {R_REF} m, degree {N_DEG}",
            (f"Regions: core r < {r_core:g} m, belt |lat| < {belt_lat:g} deg, volume fractions "
             f"{np.array2string(volumes / volumes.sum(), precision=3)}" if mode == "regions"
             else f"Mascons: {mode}, {len(pos)} mascons, {K} density groups"),
            f"Observables: {n_info} informative of {len(names)} (sigma CoM {SIGMA_COM:g} m, "
            f"sigma_CS {SIGMA_CS} + model {SIGMA_MODEL:g} in quadrature"
            f"{f', GM sigma {SIGMA_GM:g}' if USE_GM else ', no GM'}); {dof} degrees of freedom",
            f"Bulk density {rho_bulk:.3f} kg/m^3; chi2 homogeneous {r_hom @ r_hom:.6g}, "
            f"least squares {np.sum(model.residuals(rho_lsq) ** 2):.6g}, "
            f"posterior median {chi2_med:.6g}",
            f"Centre of mass (posterior median) {np.array2string(com, precision=3)} m",
            f"MCMC: {N_WALKERS} walkers x {chain.shape[0]} steps, burn-in {burn}, thin {thin}, "
            f"tau max {np.max(tau):.1f}",
        ],
    )
    print(f"[Done] {time.time() - t_start:.0f} s; outputs in {fig_dir}")
