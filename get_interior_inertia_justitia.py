r"""
Moments of inertia of 269 Justitia (EMA target) from degree-2 gravity and rotational
dynamics, by MCMC: the analysis of get_interior_inertia_psyche.py (Ermakov 2026, PSJ 7:170)
applied to Justitia, as a mission-design study with synthetic data.

No orbit propagation, no MCMCModel. The shape, two-layer model, precession, posteriors,
sampler and figures are shared with get_interior_inertia_psyche.py in Interior.py; the corner
plots reuse the house style of MCMCPlots.py (configure, plot_corner, save_figure, write_output).

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

import time
from pathlib import Path

import numpy as np

import MCMCPlots
from Interior import (
    AngularGrid,
    ci_string,
    endmember_map,
    flat_posterior,
    inertia_quantities,
    LogCoreCoordinates,
    masked_extreme,
    MeshShape,
    moment_grid_H_range,
    MomentsPosterior,
    plot_core_sections,
    plot_endmembers,
    plot_mean_moment,
    plot_sigma_I,
    plot_traces,
    results_dir,
    run_emcee,
    SolarPrecession,
    summary_row,
    TwoLayer,
    TwoLayerPosterior,
    with_mean_moment,
    write_npz,
    write_text,
)

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
# Main
# ============================================================

if __name__ == "__main__":
    t_start = time.time()
    MCMCPlots.configure(SYM_B, UNIT_B, np.ones(3), body=BODY)

    # --- 1. shape and homogeneous body ---
    mesh = MeshShape.from_obj(SHAPE_OBJ, 1e3)
    vol0, first0, _ = mesh.integrals()
    shift = first0 / vol0  # centroid of the mesh = centre of mass of the homogeneous body
    shape = MeshShape(mesh.V - shift, mesh.T)
    t0 = time.time()
    grid = AngularGrid(shape, N_MU, N_PHI)
    t_grid = time.time() - t0
    V = grid.outer[0]
    r_min = grid.R.min() / 1e3
    hom = inertia_quantities(*(RHO_MEAN * np.asarray(x) for x in grid.outer), R0, axes=True)
    exact = inertia_quantities(*(RHO_MEAN * np.asarray(x) for x in shape.integrals()), R0, axes=True)
    spin_C = np.degrees(np.arccos(abs(hom["axes"][2, 2])))  # body z (spin) vs axis of C
    print(f"[Shape] {SHAPE_OBJ.name}: {len(shape.V)} vertices, {len(shape.T)} faces, centroid shift "
          f"{np.linalg.norm(shift) / 1e3:.3f} km; {grid.w.size} nodes ({t_grid:.2f} s), V = {V:.5e} m^3 "
          f"(exact {vol0:.5e}), R_v = {grid.r_v / 1e3:.3f} km, r_min = {r_min:.2f} km, "
          f"r_max = {grid.R.max() / 1e3:.2f} km")
    print("[Homogeneous] " + ", ".join(f"{k} = {hom[k]:.4f} (exact {exact[k]:.4f})"
                                       for k in ("A", "B", "C", "J2", "C22", "H", "I"))
          + f"; spin axis vs C axis {spin_C:.2f} deg")
    model = TwoLayer(grid, RHO_MEAN, R0, axes=True)
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
    H_grid = moment_grid_H_range(hom, PRIOR_FRAC)
    print(f"[Endmember A] rho1 >= {RHO1_MIN:.0f}: H in [{minH_A:.4f}, {maxH_A:.4f}], "
          f"min I = {minI_A:.4f} ({minI_Ar:.4f} with rho2 <= {RHO2_ROCK:.0f})")
    print(f"[Endmember B] max |H - H_hom| = {dH_B:.1e}, min J2 = {minJ2_B:.4f}, min I = {minI_B:.4f}")
    print(f"[Grid 75-110 %] H in [{H_grid[0]:.3f}, {H_grid[1]:.3f}]")

    # --- 5. precession of the homogeneous body ---
    precession = SolarPrecession(POLE_RA, POLE_DEC, SPIN_RATE, ORBIT)
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
    post_B_model = TwoLayerPosterior(model, obs_B, sig_B, PRIOR_B_LO, PRIOR_B_HI, RHO1_MIN)

    fig_dir = results_dir(__file__)

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
            ch, _, tau, _ = run_emcee(MomentsPosterior(lo_A, hi_A, obs, sig), N_WALKERS, N_STEPS_A, rng,
                                      f"(a) sigma_H/H = {'none' if s is None else f'{100 * s:.1f} %'}")
            chains_A.append(ch.astype(np.float32))
            taus_A.append(tau)
        chain_B, logp_B, tau_B, map_B = run_emcee(post_B_model, N_WALKERS, N_STEPS_B, rng, "(b) two-layer",
                                                   coords=LogCoreCoordinates(post_B_model, RHO_MEAN))
        write_npz(
            fig_dir / "chain.npz", chains_A=np.array(chains_A, dtype=object), taus_A=np.array(taus_A, dtype=object),
            sweep=np.array(sweep, dtype=object), chain_B=chain_B.astype(np.float32),
            logp_B=logp_B.astype(np.float32), tau_B=tau_B, map_B=map_B, truth=TRUTH,
            obs_B=np.array(obs_B, dtype=object), sig_B=np.array(sig_B, dtype=object), noise=z,
            prior_A=np.array([lo_A, hi_A]), prior_B=np.array([PRIOR_B_LO, PRIOR_B_HI]),
            shape=SHAPE_OBJ.name, n_mu=N_MU, n_phi=N_PHI, rho_mean=RHO_MEAN, rho1_min=RHO1_MIN,
        )

    # --- posteriors (a) ---
    flat_A = [with_mean_moment(flat_posterior(ch.astype(float), tau, BURN_IN, THIN)[0])
              for ch, tau in zip(chains_A, taus_A)]
    sig_I = np.array([np.std(x[:, 3]) for x in flat_A[:-1]])
    sig_I_noH = np.std(flat_A[-1][:, 3])
    k_ref = int(np.argmin(np.abs(np.array(SIGMA_H_SWEEP) - SIGMA_H_REL)))
    for s, x in zip(sweep, flat_A):
        lab = "none" if s is None else f"{100 * s:4.1f} %"
        eq13 = "" if s is None else f", Eq. 13: {d_true['C'] * s:.4f}"
        print(f"[Posterior a] sigma_H/H {lab}: I = {ci_string(x[:, 3])}, sigma_I = {np.std(x[:, 3]):.4f}{eq13}")
    sigH_needed = 0.01 / np.array([PRIOR_FRAC[0] * hom["C"], hom["C"], PRIOR_FRAC[1] * hom["C"], d_true["C"]])
    sigH_mcmc = np.interp(0.01, sig_I, SIGMA_H_SWEEP)
    prior_A = with_mean_moment(MomentsPosterior(lo_A, hi_A, *observe(None)).sample_prior(N_PRIOR,
                                                                                     np.random.default_rng(4)))

    # --- posterior (b) ---
    post_B, burn_B, thin_B = flat_posterior(chain_B, tau_B, BURN_IN, THIN)
    sub = post_B[np.random.default_rng(3).choice(len(post_B), min(N_DERIVED, len(post_B)), replace=False)]
    D_post = np.array([post_B_model.derived(th) for th in sub])
    prior_B = post_B_model.sample_prior(N_PRIOR // 4, np.random.default_rng(5))
    D_prior = np.array([post_B_model.derived(th) for th in prior_B])
    D_true = post_B_model.derived(TRUTH)
    print(f"[Posterior b] {len(post_B)} samples (burn-in {burn_B}, thin {thin_B}); "
          + ", ".join(f"{n} = {ci_string(D_post[:, i])}" for i, n in enumerate(DER_NAMES[:4])))
    prec_B = precession(D_post[:, 2])

    # --- figures ---
    plot_endmembers(r2s, rho2s, maps, r_min, [(minI_A, kI_A), (minI_B, kI_B)], RHO1_MIN, RHO_MEAN)
    plot_sigma_I(SIGMA_H_SWEEP, sig_I, sig_I_noH, d_true["C"], hom["C"], PRIOR_FRAC)
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
    plot_core_sections(shape, model, sub, TRUTH, f"Surface ({SHAPE_OBJ.name})".replace("_", r"\_"))

    # --- summary ---
    head = f"  {'Quantity':<46s} {'Reference':>18s} {'This script':>22s}"
    lines = [
        f"269 Justitia moments of inertia (analysis of Ermakov 2026, PSJ 7:170): {Path(__file__).name}",
        f"Shape {SHAPE_OBJ.name}: {len(shape.V)} vertices, {len(shape.T)} faces, recentred by "
        f"{np.linalg.norm(shift) / 1e3:.3f} km; quadrature {N_MU} x {N_PHI}; R0 = {R0 / 1e3:.0f} km; "
        f"mean density {RHO_MEAN:.0f} kg/m^3 (assumed)",
        f"V = {V:.5e} m^3, R_v = {grid.r_v / 1e3:.3f} km, r_min = {r_min:.2f} km, "
        f"r_max = {grid.R.max() / 1e3:.2f} km",
        "",
        "1. Homogeneous body (reference: exact polyhedron integrals)",
        head,
        summary_row("volume [m^3]", f"{vol0:.5e}", f"{V:.5e}", "{}"),
    ]
    for k in ("A", "B", "C", "J2", "C22", "H", "I"):
        lines.append(summary_row(k + ("_bar" if k in ("A", "B", "C", "I") else ""), exact[k],
                                 f"{hom[k]:.4f} ({100 * (hom[k] / exact[k] - 1):+.3f} %)"))
    lines += [
        summary_row("spin axis (body z) vs C axis [deg]", "0 (principal-axis rotation)", spin_C, "{:.2f}"),
        "   a relaxed rotator spins about C: the offset measures the inconsistency of the shape (lightcurve",
        "   + occultation model), the IAU pole and uniform density; a rendezvous gravity field and pole fix it",
        "",
        "2. Two-layer endmembers (as Ermakov 2026, Sec. 4.1, Fig. 2)",
        head,
        summary_row("r_min, core exposure onset [km]", "-", r_min, "{:.2f}"),
        summary_row(f"(A) H range, rho1 >= {RHO1_MIN:.0f}", "-", f"{minH_A:.4f} - {maxH_A:.4f}"),
        summary_row(f"(A) min I_bar, rho2 <= {GRID_RHO2[1]:.0f}", "-", minI_A),
        summary_row(f"(A) min I_bar, rho2 <= {RHO2_ROCK:.0f}", "-", minI_Ar),
        summary_row("(A) min I_bar at r2 [km], rho2 [kg/m^3]", "-", f"{r2s[kI_A[0]]:.1f}, {rho2s[kI_A[1]]:.0f}"),
        summary_row("(B) H change vs homogeneous", "0 (exact)", f"max {dH_B:.1e}"),
        summary_row("(B) min J2", "< homogeneous", f"{minJ2_B:.4f}"),
        summary_row("(B) min I_bar", "-", minI_B),
        summary_row("H range, A, B, C at 75-110 % (C largest)", "-", f"{H_grid[0]:.3f} - {H_grid[1]:.3f}"),
        "",
        "3. MCMC (a): theta = (A_bar, B_bar, C_bar), prior 75-110 % of homogeneous",
        f"   data J2, C22 (sigma/value {100 * SIGMA_J2_REL:.3f} %) + H; truth = two-layer TRUTH; "
        f"{N_WALKERS} walkers x {N_STEPS_A} steps",
        f"  {'sigma_H/H':<12s} {'I_bar (16-50-84 %)':>30s} {'sigma_I MCMC':>14s} {'Eq. 13':>10s}",
    ]
    for s, x in zip(sweep, flat_A):
        lab = "no H" if s is None else f"{100 * s:.1f} %"
        eq = "-" if s is None else f"{d_true['C'] * s:.4f}"
        lines.append(f"  {lab:<12s} {ci_string(x[:, 3]):>30s} {np.std(x[:, 3]):>14.4f} {eq:>10s}")
    lines += [
        f"  prior only: I_bar {ci_string(prior_A[:, 3])}, sigma {np.std(prior_A[:, 3]):.4f}",
        f"  {'requirement sigma_I = 0.01':<46s} {'Eq. 13':>18s} {'MCMC (a)':>22s}",
        summary_row("sigma_H/H, C_bar 75 % of homogeneous", f"{100 * sigH_needed[0]:.2f} %", "-", "{}"),
        summary_row("sigma_H/H, C_bar homogeneous", f"{100 * sigH_needed[1]:.2f} %", "-", "{}"),
        summary_row("sigma_H/H, C_bar 110 % of homogeneous", f"{100 * sigH_needed[2]:.2f} %", "-", "{}"),
        summary_row("sigma_H/H, truth C_bar", f"{100 * sigH_needed[3]:.2f} %", f"{100 * sigH_mcmc:.2f} %", "{}"),
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
        lines.append(f"  {n:<10s} {TRUTH[i]:>12.4f} {ci_string(post_B[:, i]):>34s} {ci_string(prior_B[:, i]):>34s}")
    for i, n in enumerate(DER_NAMES):
        lines.append(f"  {n:<10s} {D_true[i]:>12.4f} {ci_string(D_post[:, i]):>34s} {ci_string(D_prior[:, i]):>34s}")
    rate_req = sigH_needed[1] * prec["rate"]
    lines += [
        "",
        "5. Precession (Ermakov 2026, Eq. 1), homogeneous body",
        f"   pole RA {POLE_RA:.4f}, Dec {POLE_DEC:.4f} deg, spin {SPIN_RATE} deg/day "
        f"(P = {360.0 / SPIN_RATE * 24:.2f} h); orbit a {ORBIT['a']} AU, e {ORBIT['e']}, "
        f"i {ORBIT['i']} deg, node {ORBIT['node']} deg (osculating, 2034-01-01)",
        summary_row("obliquity [deg]", "-", prec["eps"], "{:.2f}"),
        summary_row("precession period [My]", "-", prec["period"], "{:.3f}"),
        summary_row("d alpha / dt [mas/yr]", "-", prec["dra"], "{:.1f}"),
        summary_row("d delta / dt [mas/yr]", "-", prec["ddec"], "{:.1f}"),
        summary_row("pole drift rate [mas/yr]", "-", prec["rate"], "{:.1f}"),
        summary_row("pole drift rate, MCMC (b) H [mas/yr]", "-",
                    f"{np.median(prec_B['rate']):.1f} +/- {np.std(prec_B['rate']):.1f}", "{}"),
        f"   the pole drift rate is proportional to H: sigma_H / H = {100 * SIGMA_H_REL:.1f} % <-> "
        f"{SIGMA_H_REL * prec['rate']:.1f} mas/yr; sigma_I = 0.01 (C_bar homogeneous) needs "
        f"sigma_H / H = {100 * sigH_needed[1]:.2f} % <-> {rate_req:.1f} mas/yr",
        f"   ({rate_req * 10:.0f} mas = {np.radians(rate_req * 10 / 3.6e6) * grid.r_v:.2f} m on the "
        f"surface over a 10-yr baseline)",
    ]
    write_text(fig_dir / "summary.txt", lines)
    print(f"[Done] {time.time() - t_start:.0f} s; outputs in {fig_dir}")
