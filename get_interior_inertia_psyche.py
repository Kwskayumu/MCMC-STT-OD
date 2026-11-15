r"""
Moments of inertia of 16 Psyche from degree-2 gravity and rotational dynamics, by MCMC,
compared with Ermakov (2026), PSJ 7:170, "Interrogating 16 Psyche's Interior with
Rotational Dynamics and Gravity Field".

No orbit propagation, no MCMCModel. The shape, two-layer model, precession, posteriors,
sampler and figures are shared with get_interior_inertia_justitia.py in Interior.py; the corner
plots reuse the house style of MCMCPlots.py (configure, plot_corner, save_figure, write_output).

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
from pathlib import Path

import numpy as np

import MCMCPlots
from Interior import (
    AngularGrid,
    LogCoreCoordinates,
    MomentsPosterior,
    SHShape,
    SolarPrecession,
    TwoLayer,
    TwoLayerPosterior,
    ci_string,
    endmember_map,
    flat_posterior,
    inertia_quantities,
    masked_extreme,
    moment_grid_H_range,
    plot_core_sections,
    plot_endmembers,
    plot_mean_moment,
    plot_sigma_I,
    plot_traces,
    run_emcee,
    summary_row,
    with_mean_moment,
)

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
# Main
# ============================================================

if __name__ == "__main__":
    t_start = time.time()
    MCMCPlots.configure(SYM_B, UNIT_B, np.ones(3), body=BODY)

    # --- 1. shape and homogeneous body ---
    shape = SHShape.from_file(SHAPE_SH)
    grid = AngularGrid(shape, N_MU, N_PHI)
    V = grid.outer[0]
    r_min = grid.R.min() / 1e3
    hom = inertia_quantities(*(RHO_MEAN * np.asarray(x) for x in grid.outer), R0)
    print(f"[Shape] {SHAPE_SH.name}: degree {shape.degree}, {grid.w.size} nodes, "
          f"V = {V:.5e} m^3, R_v = {grid.r_v / 1e3:.3f} km, r_min = {r_min:.2f} km, "
          f"r_max = {grid.R.max() / 1e3:.2f} km, CoM offset {np.linalg.norm(hom['com']) / 1e3:.2f} km")
    print("[Homogeneous] " + ", ".join(f"{k} = {hom[k]:.4f} (paper {PAPER_HOM[k]:.4f})" for k in PAPER_HOM)
          + f", I = {hom['I']:.4f}")
    model = TwoLayer(grid, RHO_MEAN, R0)
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
    H_grid = moment_grid_H_range(hom, PRIOR_FRAC)
    print(f"[Endmember A] rho1 >= {RHO1_MIN:.0f}: min H = {minH_A:.4f} (paper {PAPER_MIN_H}), "
          f"min I = {minI_A:.4f} (paper {PAPER_MIN_I}; {minI_A8:.4f} with rho2 <= {RHO2_METAL:.0f})")
    print(f"[Endmember B] max |H - H_hom| = {dH_B:.1e}, min J2 = {minJ2_B:.4f}, min I = {minI_B:.4f}")
    print(f"[Grid 75-110 %] H in [{H_grid[0]:.3f}, {H_grid[1]:.3f}] (paper {PAPER_H_RANGE})")

    # --- 5. precession of the homogeneous body ---
    precession = SolarPrecession(POLE_RA, POLE_DEC, SPIN_RATE, ORBIT)
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
    post_B_model = TwoLayerPosterior(model, obs_B, sig_B, PRIOR_B_LO, PRIOR_B_HI, RHO1_MIN)

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
            ch, _, tau, _ = run_emcee(MomentsPosterior(lo_A, hi_A, obs, sig), N_WALKERS, N_STEPS_A, rng,
                                      f"(a) sigma_H/H = {'none' if s is None else f'{100 * s:.1f} %'}")
            chains_A.append(ch.astype(np.float32))
            taus_A.append(tau)
        chain_B, logp_B, tau_B, map_B = run_emcee(post_B_model, N_WALKERS, N_STEPS_B, rng, "(b) two-layer",
                                                   coords=LogCoreCoordinates(post_B_model, RHO_MEAN))
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
    flat_A = [with_mean_moment(flat_posterior(ch.astype(float), tau, BURN_IN, THIN)[0])
              for ch, tau in zip(chains_A, taus_A)]
    sig_I = np.array([np.std(x[:, 3]) for x in flat_A[:-1]])
    sig_I_noH = np.std(flat_A[-1][:, 3])
    k18 = int(np.argmin(np.abs(np.array(SIGMA_H_SWEEP) - SIGMA_H_REL)))
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
    plot_endmembers(r2s, rho2s, maps, r_min, [(minI_A, kI_A), (minI_B, kI_B)], RHO1_MIN, RHO_MEAN,
                    levels=dict(H=np.arange(0.18, 0.30, 0.02), J2=np.arange(0.06, 0.16, 0.02),
                                rho1=[2000, 3000, 4000]), fmt_J2="%.2f")
    plot_sigma_I(SIGMA_H_SWEEP, sig_I, sig_I_noH, d_true["C"], hom["C"], PRIOR_FRAC,
                 paper_sigma_H=PAPER_SIGMA_H_REL)
    plot_mean_moment(
        [
            (prior_A[:, 3], "Prior (a)", MCMCPlots.C_PRIOR, "--", False),
            (flat_A[-1][:, 3], r"MCMC (a), $J_2$, $C_{22}$", "0.45", "-", False),
            (flat_A[k18][:, 3], r"MCMC (a), $+ H$ (%.1f$\%%$)" % (100 * SIGMA_H_REL), MCMCPlots.C_POST, "-", True),
            (D_post[:, 0], "MCMC (b), Two-Layer", MCMCPlots.C_SC, "-", False),
        ],
        d_true["I"], hom["I"], PAPER_MIN_I, min_label=r"Paper Minimum (Two-Layer)",
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
    plot_core_sections(shape, model, sub, TRUTH, "Surface (Shepard et al. 2021)")

    # --- summary: paper vs this script ---
    head = f"  {'Quantity':<46s} {'Ermakov (2026)':>18s} {'This script':>22s}"
    lines = [
        f"16 Psyche moments of inertia: Ermakov (2026, PSJ 7:170) vs {Path(__file__).name}",
        f"Shape {SHAPE_SH.name} (Zenodo 10.5281/zenodo.12522382), degree {shape.degree}; "
        f"quadrature {N_MU} x {N_PHI}; R0 = {R0 / 1e3:.0f} km; mean density {RHO_MEAN:.0f} kg/m^3",
        f"V = {V:.5e} m^3, R_v = {grid.r_v / 1e3:.3f} km, centre-of-mass offset "
        f"{np.linalg.norm(hom['com']) / 1e3:.3f} km",
        "",
        "1. Homogeneous body (paper Table 1, Shepard et al. 2021 row)",
        head,
    ]
    for k in PAPER_HOM:
        lines.append(summary_row(f"{k}" + ("" if k in ("J2", "C22", "H") else "_bar"), PAPER_HOM[k],
                                 f"{hom[k]:.4f} ({100 * (hom[k] / PAPER_HOM[k] - 1):+.2f} %)"))
    I_paper = PAPER_HOM["J2"] * (1 / PAPER_HOM["H"] - 2 / 3)
    lines += [
        summary_row("I_bar = J2 (1/H - 2/3)", I_paper, hom["I"]),
        "",
        "2. Two-layer endmembers (paper Sec. 4.1, Fig. 2)",
        head,
        summary_row("r_min, core exposure onset [km]", PAPER_RMIN, r_min, "{:.1f}"),
        summary_row(f"(A) min H, rho1 >= {RHO1_MIN:.0f}", PAPER_MIN_H, minH_A),
        summary_row(f"(A) min I_bar, rho2 <= {GRID_RHO2[1]:.0f}", PAPER_MIN_I, minI_A),
        summary_row(f"(A) min I_bar, rho2 <= {RHO2_METAL:.0f}", "-", minI_A8),
        summary_row("(A) min I_bar at r2 [km], rho2 [kg/m^3]", "-",
                    f"{r2s[kI_A[0]]:.1f}, {rho2s[kI_A[1]]:.0f}"),
        summary_row("(B) H change vs homogeneous", "unaffected", f"max {dH_B:.1e}"),
        summary_row("(B) J2", "< homogeneous", f">= {minJ2_B:.4f}"),
        summary_row("H range, A, B, C at 75-110 % (C largest)", f"{PAPER_H_RANGE[0]:.2f} - {PAPER_H_RANGE[1]:.2f}",
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
        lines.append(f"  {lab:<12s} {ci_string(x[:, 3]):>30s} {np.std(x[:, 3]):>14.4f} {eq:>10s}")
    lines += [
        f"  prior only: I_bar {ci_string(prior_A[:, 3])}, sigma {np.std(prior_A[:, 3]):.4f}",
        head,
        summary_row("sigma_H/H for sigma_I = 0.01, C_bar 75 % hom", f"{PAPER_SIGMA_H_REL[1]:.1f} %",
                    f"{100 * sigH_needed[0]:.2f} % (Eq. 13)", "{}"),
        summary_row("sigma_H/H for sigma_I = 0.01, C_bar homogeneous", f"{PAPER_SIGMA_H_REL[0]:.1f} %",
                    f"{100 * sigH_needed[1]:.2f} % (Eq. 13)", "{}"),
        summary_row("sigma_H/H for sigma_I = 0.01, C_bar 110 % hom", "-",
                    f"{100 * sigH_needed[2]:.2f} % (Eq. 13)", "{}"),
        summary_row("sigma_H/H for sigma_I = 0.01, truth C_bar", "-",
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
        lines.append(f"  {n:<10s} {TRUTH[i]:>12.4f} {ci_string(post_B[:, i]):>34s} {ci_string(prior_B[:, i]):>34s}")
    for i, n in enumerate(DER_NAMES):
        lines.append(f"  {n:<10s} {D_true[i]:>12.4f} {ci_string(D_post[:, i]):>34s} {ci_string(D_prior[:, i]):>34s}")
    lines += [
        "",
        "5. Precession (paper Eq. 1, Table 3), homogeneous body",
        f"   pole RA {POLE_RA}, Dec {POLE_DEC} deg; orbit a {ORBIT['a']} AU, e {ORBIT['e']}, "
        f"i {ORBIT['i']} deg, node {ORBIT['node']} deg (approximate)",
        head,
        summary_row("obliquity [deg]", f"{PAPER_OBLIQUITY[0]} - {PAPER_OBLIQUITY[1]} (10 My)", prec["eps"], "{:.2f}"),
        summary_row("precession period, present obliquity [My]", "-", prec["period"], "{:.2f}"),
        summary_row(f"precession period, mean obliquity {eps_mean:.1f} deg [My]", PAPER_PREC_PERIOD,
                    period_mean, "{:.2f}"),
        summary_row("d alpha / dt [mas/yr]", PAPER_POLE_RATE[0], prec["dra"], "{:.1f}"),
        summary_row("d delta / dt [mas/yr]", PAPER_POLE_RATE[1], prec["ddec"], "{:.1f}"),
        summary_row("pole drift rate, MCMC (b) H [mas/yr]", "-",
                    f"{np.median(prec_B['rate']):.1f} +/- {np.std(prec_B['rate']):.1f}", "{}"),
        f"   pole drift rate is proportional to H (paper Eq. 12): sigma_H / H = "
        f"{100 * SIGMA_H_REL:.1f} % <-> {SIGMA_H_REL * prec['rate']:.1f} mas/yr on "
        f"{prec['rate']:.1f} mas/yr",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    MCMCPlots.write_output(fig_dir / "summary.txt", text.encode(), tag="Summary")
    print(f"[Done] {time.time() - t_start:.0f} s; outputs in {fig_dir}")
