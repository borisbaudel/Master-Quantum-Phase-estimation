import sys
import os
import json
import csv
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from joblib import Parallel, delayed

from core.kalman import DiscreteKalmanFilter
from core.model import (
    build_Aa,
    build_Ca,
    build_continuous_process_covariance,
    discretize_system_van_loan as discretize_plant,
)
from core.qco_model import (
    build_qco_augmented_A,
    build_qco_measurement,
    build_qco_process_covariance,
    discretize_system_van_loan as discretize_qco,
)

# =========================================================
# Global constants
# =========================================================
EPS = 1e-12
_DEFAULT_C_PS = 1.0
_DEFAULT_C_PO = 0.4

# =========================================================
# Plot style
# =========================================================
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.titlesize": 15,
    "lines.linewidth": 1.8,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linestyle": "-",
    "axes.spines.top": True,
    "axes.spines.right": True,
})

# =========================================================
# Reproducibility config
# =========================================================
@dataclass
class SimulationConfig:
    mode: str = "FINAL_HEAVY"

    # Sampling / trajectory
    dt: float = 0.02
    T: float = 40.0
    burn_fraction: float = 0.20

    # Monte Carlo
    n_mc: int = 5
    global_seed: int = 20260328

    # Plant parameters
    gamma: float = 1.0
    omega: float = 0.0
    g_phi: float = 2.0
    kappa: float = 1.0
    sigma_q: float = 0.02
    sigma_p: float = 0.02
    q_phi: float = 0.01
    meas_std: float = 0.03

    # QCO parameters
    gamma_s: float = 1.0
    omega_s: float = 0.0
    gamma_o: float = 1.2
    omega_o: float = 0.4
    sigma_qs: float = 0.02
    sigma_ps: float = 0.02
    sigma_qo: float = 0.02
    sigma_po: float = 0.02
    c_qs: float = 0.0
    c_ps: float = _DEFAULT_C_PS
    c_qo: float = 0.0
    c_po: float = _DEFAULT_C_PO
    k_so: float = 0.8
    k_os: float = 0.8

    # Gramian horizon proxy for discrete simulation metric
    gramian_n_decay: float = 5.0
    gramian_floor: int = 20
    gramian_cap: int = 200

    # Grids
    theta_min: float = 0.05
    theta_max: float = np.pi - 0.05
    n_theta: int = 48

    lambda_min: float = 0.02
    lambda_max: float = 1.50
    n_lambda: int = 36

    # Plot switches
    do_plant_families: bool = True
    do_plant_global_noncollapse: bool = True
    do_qco_map: bool = True
    do_qco_performance_scatter: bool = True
    do_tradeoff_scatter: bool = True

    # Global non-collapse rendering
    noncollapse_use_hexbin: bool = True
    noncollapse_max_points: int = 1400

    # Parallel
    n_jobs: int = max(1, (os.cpu_count() or 4) - 1)

    # Output
    save_dir: str = "plots_paper_ready"
    save_raw_csv: bool = True
    save_config_json: bool = True


# =========================================================
# Utilities
# =========================================================
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def stable_int_seed(*args, base_seed=0) -> int:
    key = "|".join([str(a) for a in args]) + f"|base={base_seed}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**31)


def gramian_horizon(gamma: float, dt: float,
                    n_decay: float = 5.0,
                    floor: int = 20,
                    cap: int = 200) -> int:
    steps = int(np.ceil(n_decay / max(gamma * dt, 1e-12)))
    return max(floor, min(steps, cap))


def safe_relative_error_improvement(err_base: float, err_new: float,
                                    eps: float = 1e-12) -> float:
    denom = max(abs(err_base), eps)
    return float((err_base - err_new) / denom)


def log_gain(a: float, b: float, eps: float = 1e-12) -> float:
    return float(np.log(max(b, eps)) - np.log(max(a, eps)))


def relative_gain(a: float, b: float, eps: float = 1e-12) -> float:
    denom = max(abs(a), eps)
    return float((b - a) / denom)


def robust_sym_limits(values, q=0.995, min_abs=1e-3):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -1.0, 1.0
    vmax = np.quantile(np.abs(arr), q)
    vmax = max(float(vmax), min_abs)
    return -vmax, vmax


def export_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_csv_dicts(rows, path):
    if len(rows) == 0:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =========================================================
# Metrics
# =========================================================
def phase_observability_metric(F: np.ndarray, H: np.ndarray,
                               phase_index: int, horizon: int) -> float:
    n = F.shape[0]
    e_phi = np.zeros(n, dtype=float)
    e_phi[phase_index] = 1.0

    Fk = np.eye(n)
    J_phi = 0.0
    for _ in range(horizon):
        out = H @ (Fk @ e_phi)
        out = np.asarray(out, dtype=float).ravel()
        J_phi += float(np.dot(out, out))
        Fk = F @ Fk
    return J_phi


def phase_projected_noise_metric(Qd: np.ndarray, phase_index: int) -> float:
    return float(Qd[phase_index, phase_index])


def output_noise_metric(F: np.ndarray, H: np.ndarray, Qd: np.ndarray,
                        horizon: int) -> float:
    Fk = np.eye(F.shape[0], dtype=float)
    total = 0.0
    for _ in range(horizon):
        S_k = H @ Fk @ Qd @ Fk.T @ H.T
        total += float(np.trace(S_k))
        Fk = F @ Fk
    return total


# =========================================================
# Simulation core
# =========================================================
def simulate_kf_rmse(
    F, H, Qd, R,
    x_true0, x_hat0, P0,
    n_steps, burn,
    phase_index,
    seed,
):
    rng = np.random.default_rng(seed)

    n_state = F.shape[0]
    chol_Q = np.linalg.cholesky(Qd + 1e-15 * np.eye(n_state))
    meas_std = float(np.sqrt(R[0, 0]))

    x_true = x_true0.copy()
    kf = DiscreteKalmanFilter(
        F=F, H=H, Qd=Qd, R=R,
        x0=x_hat0.copy(), P0=P0.copy()
    )

    se = 0.0
    count = 0

    for k in range(n_steps):
        w_k = chol_Q @ rng.standard_normal(n_state)
        x_true = F @ x_true + w_k

        y_k = float((H @ x_true).item() + meas_std * rng.standard_normal())
        x_est = kf.step(np.array([y_k]))

        if k >= burn:
            err = x_true[phase_index] - x_est[phase_index]
            se += float(err * err)
            count += 1

    return float(np.sqrt(se / max(count, 1)))


def mc_mean_rmse(sim_fn, n_mc: int, seeds):
    vals = [sim_fn(s) for s in seeds]
    return float(np.mean(vals)), float(np.std(vals, ddof=0))


# =========================================================
# Model builders
# =========================================================
def build_plant_discrete(theta, lam, cfg: SimulationConfig):
    A_p = build_Aa(
        gamma=cfg.gamma,
        omega=cfg.omega,
        g_phi=cfg.g_phi,
        lam=lam,
    )
    C_p = build_Ca(kappa=cfg.kappa, theta=theta)
    Qc_p = build_continuous_process_covariance(
        sigma_q=cfg.sigma_q,
        sigma_p=cfg.sigma_p,
        q_phi=cfg.q_phi,
    )
    F_p, Qd_p = discretize_plant(A_p, Qc_p, cfg.dt)
    return F_p, C_p, Qd_p


def build_qco_discrete(theta, lam, cfg: SimulationConfig):
    A_q = build_qco_augmented_A(
        gamma_s=cfg.gamma_s, omega_s=cfg.omega_s,
        gamma_o=cfg.gamma_o, omega_o=cfg.omega_o,
        g_phi=cfg.g_phi, lam=lam,
        k_so=cfg.k_so, k_os=cfg.k_os,
    )
    H_q = build_qco_measurement(
        c_qs=cfg.c_qs,
        c_ps=cfg.c_ps,
        c_qo=cfg.c_qo,
        c_po=cfg.c_po,
    )
    Qc_q = build_qco_process_covariance(
        sigma_qs=cfg.sigma_qs,
        sigma_ps=cfg.sigma_ps,
        sigma_qo=cfg.sigma_qo,
        sigma_po=cfg.sigma_po,
        q_phi=cfg.q_phi,
    )
    F_q, Qd_q = discretize_qco(A_q, Qc_q, cfg.dt)
    return F_q, H_q, Qd_q


# =========================================================
# Plant metrics
# =========================================================
def plant_metrics(theta, lam, cfg: SimulationConfig):
    F_p, H_p, Qd_p = build_plant_discrete(theta, lam, cfg)
    R = np.array([[cfg.meas_std ** 2]], dtype=float)

    n_steps = int(cfg.T / cfg.dt)
    burn = int(cfg.burn_fraction * n_steps)
    horizon = gramian_horizon(
        gamma=cfg.gamma,
        dt=cfg.dt,
        n_decay=cfg.gramian_n_decay,
        floor=cfg.gramian_floor,
        cap=cfg.gramian_cap,
    )

    x_true0 = np.array([0.0, 0.0, 0.5], dtype=float)
    x_hat0 = np.zeros(3, dtype=float)
    P0 = 10.0 * np.eye(3)

    seeds = [
        stable_int_seed("plant", theta, lam, i, base_seed=cfg.global_seed)
        for i in range(cfg.n_mc)
    ]

    def _one_run(seed):
        return simulate_kf_rmse(
            F=F_p, H=H_p, Qd=Qd_p, R=R,
            x_true0=x_true0, x_hat0=x_hat0, P0=P0,
            n_steps=n_steps, burn=burn,
            phase_index=2, seed=seed,
        )

    mean_rmse, std_rmse = mc_mean_rmse(_one_run, cfg.n_mc, seeds)
    J_phi = phase_observability_metric(F_p, H_p, phase_index=2, horizon=horizon)
    noise_phi = phase_projected_noise_metric(Qd_p, phase_index=2)
    noise_out = output_noise_metric(F_p, H_p, Qd_p, horizon=horizon)

    return {
        "theta": float(theta),
        "lambda": float(lam),
        "rmse_phi": mean_rmse,
        "rmse_phi_std_mc": std_rmse,
        "J_phi": J_phi,
        "inv_sqrt_Jphi": 1.0 / np.sqrt(max(J_phi, EPS)),
        "noise_phi": noise_phi,
        "noise_out": noise_out,
        "N_gramian": int(horizon),
        "n_steps": int(n_steps),
        "burn_steps": int(burn),
        "seeds": seeds,
        "F": F_p,
        "H": H_p,
        "Qd": Qd_p,
        "R": R,
    }


# =========================================================
# QCO metrics
# =========================================================
def qco_metrics(theta, lam, cfg: SimulationConfig):
    F_q, H_q, Qd_q = build_qco_discrete(theta, lam, cfg)
    R = np.array([[cfg.meas_std ** 2]], dtype=float)

    n_steps = int(cfg.T / cfg.dt)
    burn = int(cfg.burn_fraction * n_steps)
    horizon = gramian_horizon(
        gamma=cfg.gamma_s,
        dt=cfg.dt,
        n_decay=cfg.gramian_n_decay,
        floor=cfg.gramian_floor,
        cap=cfg.gramian_cap,
    )

    x_true0 = np.array([0.0, 0.0, 0.0, 0.0, 0.5], dtype=float)
    x_hat0 = np.zeros(5, dtype=float)
    P0 = 10.0 * np.eye(5)

    seeds = [
        stable_int_seed("qco", theta, lam, i, base_seed=cfg.global_seed)
        for i in range(cfg.n_mc)
    ]

    def _one_run(seed):
        return simulate_kf_rmse(
            F=F_q, H=H_q, Qd=Qd_q, R=R,
            x_true0=x_true0, x_hat0=x_hat0, P0=P0,
            n_steps=n_steps, burn=burn,
            phase_index=4, seed=seed,
        )

    mean_rmse, std_rmse = mc_mean_rmse(_one_run, cfg.n_mc, seeds)
    J_phi = phase_observability_metric(F_q, H_q, phase_index=4, horizon=horizon)
    noise_phi = phase_projected_noise_metric(Qd_q, phase_index=4)
    noise_out = output_noise_metric(F_q, H_q, Qd_q, horizon=horizon)

    return {
        "theta": float(theta),
        "lambda": float(lam),
        "rmse_phi": mean_rmse,
        "rmse_phi_std_mc": std_rmse,
        "J_phi": J_phi,
        "inv_sqrt_Jphi": 1.0 / np.sqrt(max(J_phi, EPS)),
        "noise_phi": noise_phi,
        "noise_out": noise_out,
        "N_gramian": int(horizon),
        "n_steps": int(n_steps),
        "burn_steps": int(burn),
        "seeds": seeds,
        "F": F_q,
        "H": H_q,
        "Qd": Qd_q,
        "R": R,
    }


# =========================================================
# Grid collection
# =========================================================
def collect_plant_grid(theta_vals, lambda_vals, cfg: SimulationConfig):
    tasks = [(i, j, lam, theta)
             for i, lam in enumerate(lambda_vals)
             for j, theta in enumerate(theta_vals)]

    def _run(i, j, lam, theta):
        met = plant_metrics(theta, lam, cfg)
        return i, j, met

    results = Parallel(n_jobs=cfg.n_jobs, backend="loky")(
        delayed(_run)(i, j, lam, theta) for i, j, lam, theta in tasks
    )

    shape = (len(lambda_vals), len(theta_vals))
    rmse_map = np.zeros(shape)
    jphi_map = np.zeros(shape)
    invsqrt_map = np.zeros(shape)
    noise_out_map = np.zeros(shape)

    rows = []
    lookup = {}

    for i, j, met in results:
        rmse_map[i, j] = met["rmse_phi"]
        jphi_map[i, j] = met["J_phi"]
        invsqrt_map[i, j] = met["inv_sqrt_Jphi"]
        noise_out_map[i, j] = met["noise_out"]

        row = {
            "theta": met["theta"],
            "lambda": met["lambda"],
            "rmse_phi": met["rmse_phi"],
            "rmse_phi_std_mc": met["rmse_phi_std_mc"],
            "J_phi": met["J_phi"],
            "inv_sqrt_Jphi": met["inv_sqrt_Jphi"],
            "noise_out": met["noise_out"],
            "N_gramian": met["N_gramian"],
            "n_steps": met["n_steps"],
            "burn_steps": met["burn_steps"],
        }
        rows.append(row)
        lookup[(i, j)] = met

    return {
        "theta_vals": theta_vals,
        "lambda_vals": lambda_vals,
        "rmse_map": rmse_map,
        "jphi_map": jphi_map,
        "invsqrt_map": invsqrt_map,
        "noise_out_map": noise_out_map,
        "rows": rows,
        "lookup": lookup,
    }


def compare_plant_vs_qco(plant_met, theta, lam, cfg: SimulationConfig):
    qco_met = qco_metrics(theta, lam, cfg)

    delta_rmse = safe_relative_error_improvement(
        plant_met["rmse_phi"], qco_met["rmse_phi"]
    )
    delta_jphi_rel = relative_gain(plant_met["J_phi"], qco_met["J_phi"])
    delta_jphi_log = log_gain(plant_met["J_phi"], qco_met["J_phi"])
    delta_qphi = relative_gain(plant_met["noise_phi"], qco_met["noise_phi"])
    delta_qout = relative_gain(plant_met["noise_out"], qco_met["noise_out"])

    return {
        "theta": float(theta),
        "lambda": float(lam),
        "rmse_plant": plant_met["rmse_phi"],
        "rmse_qco": qco_met["rmse_phi"],
        "delta_rmse": float(delta_rmse),
        "delta_Jphi_rel": float(delta_jphi_rel),
        "delta_Jphi_log": float(delta_jphi_log),
        "delta_Qphi": float(delta_qphi),
        "delta_Qout": float(delta_qout),
        "Jphi_plant": float(plant_met["J_phi"]),
        "Jphi_qco": float(qco_met["J_phi"]),
    }


def collect_qco_comparison_grid(plant_data, cfg: SimulationConfig):
    theta_vals = plant_data["theta_vals"]
    lambda_vals = plant_data["lambda_vals"]
    plant_lookup = plant_data["lookup"]

    tasks = [(i, j, lambda_vals[i], theta_vals[j], plant_lookup[(i, j)])
             for i in range(len(lambda_vals))
             for j in range(len(theta_vals))]

    def _run(i, j, lam, theta, plant_met):
        comp = compare_plant_vs_qco(plant_met, theta, lam, cfg)
        return i, j, comp

    results = Parallel(n_jobs=cfg.n_jobs, backend="loky")(
        delayed(_run)(i, j, lam, theta, plant_met)
        for i, j, lam, theta, plant_met in tasks
    )

    shape = (len(lambda_vals), len(theta_vals))
    delta_rmse_map = np.zeros(shape)
    delta_jphi_log_map = np.zeros(shape)
    delta_qout_map = np.zeros(shape)

    rows = []

    for i, j, comp in results:
        delta_rmse_map[i, j] = comp["delta_rmse"]
        delta_jphi_log_map[i, j] = comp["delta_Jphi_log"]
        delta_qout_map[i, j] = comp["delta_Qout"]
        rows.append(comp)

    return {
        "theta_vals": theta_vals,
        "lambda_vals": lambda_vals,
        "delta_rmse_map": delta_rmse_map,
        "delta_jphi_log_map": delta_jphi_log_map,
        "delta_qout_map": delta_qout_map,
        "rows": rows,
    }


# =========================================================
# Paper-ready plotting helpers
# =========================================================
def finalize_figure(fig, path_png, path_pdf=None):
    fig.tight_layout()
    fig.savefig(path_png, dpi=350, bbox_inches="tight")
    if path_pdf is not None:
        fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(X, Y, Z, xlabel, ylabel, clabel, title, out_base,
                 cmap="viridis", center_zero=False):
    fig, ax = plt.subplots(figsize=(7.0, 4.9))

    if center_zero:
        vmin, vmax = robust_sym_limits(Z)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = None

    im = ax.pcolormesh(X, Y, Z, shading="auto", cmap=cmap, norm=norm)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label(clabel, fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    finalize_figure(fig, out_base + ".png", out_base + ".pdf")


# =========================================================
# Plot 1: global non-collapse, paper-ready
# =========================================================
def plot_global_noncollapse(data, save_dir, prefix="plant", max_points=1400, use_hexbin=False):
    rows = data["rows"]
    x = np.array([r["J_phi"] for r in rows], dtype=float)
    y = np.array([r["rmse_phi"] for r in rows], dtype=float)
    lam = np.array([r["lambda"] for r in rows], dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(lam) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    lam = lam[mask]

    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
        x_plot = x[idx]
        y_plot = y[idx]
        lam_plot = lam[idx]
    else:
        x_plot, y_plot, lam_plot = x, y, lam

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    if use_hexbin:
        hb = ax.hexbin(
            x_plot, y_plot,
            C=lam_plot,
            reduce_C_function=np.mean,
            gridsize=35,
            xscale="log",
            yscale="log",
            cmap="viridis",
            mincnt=1,
            linewidths=0.0
        )
        cbar = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.03)
    else:
        sc = ax.scatter(
            x_plot, y_plot,
            c=lam_plot,
            cmap="viridis",
            s=9,
            alpha=0.68,
            edgecolors="none",
            rasterized=True
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)

    cbar.set_label(r"$\lambda$", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Directional observability metric $J_{\phi}(T)$")
    ax.set_ylabel(r"Phase-estimation RMSE")
    ax.set_title("Global non-collapse across regimes")

    ax.grid(True, which="major", alpha=0.18)
    ax.grid(False, which="minor")

    finalize_figure(
        fig,
        os.path.join(save_dir, f"{prefix}_global_noncollapse_loglog.png"),
        os.path.join(save_dir, f"{prefix}_global_noncollapse_loglog.pdf")
    )


# =========================================================
# Plot 2: local families at fixed lambda
# =========================================================
def plot_families_fixed_lambda(data, save_dir, prefix="plant", lambda_targets=None):
    lambda_vals = data["lambda_vals"]
    rmse_map = data["rmse_map"]
    invsqrt_map = data["invsqrt_map"]

    if lambda_targets is None:
        lambda_targets = [0.02, 0.10, 0.30, 0.70, 1.50]

    indices = []
    for target in lambda_targets:
        idx = int(np.argmin(np.abs(lambda_vals - target)))
        if idx not in indices:
            indices.append(idx)

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    for idx in indices:
        lam = lambda_vals[idx]
        x = invsqrt_map[idx, :]
        y = rmse_map[idx, :]
        order = np.argsort(x)
        ax.plot(
            x[order], y[order],
            marker="o", ms=3.2,
            lw=1.6,
            label=fr"$\lambda={lam:.2f}$"
        )

    ax.set_xlabel(r"$1/\sqrt{J_{\phi}(T)}$")
    ax.set_ylabel(r"Phase-estimation RMSE")
    ax.set_title(r"Local RMSE families at fixed OU rate $\lambda$")
    ax.legend(frameon=True)

    finalize_figure(
        fig,
        os.path.join(save_dir, f"{prefix}_families_fixed_lambda.png"),
        os.path.join(save_dir, f"{prefix}_families_fixed_lambda.pdf")
    )


# =========================================================
# Plot 3: QCO performance vs phase-information gain
# =========================================================
def binned_statistics_logx(x, y, n_bins=18):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return None

    edges = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])

    means = np.full(n_bins, np.nan)
    stds = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if np.any(m):
            means[i] = np.mean(y[m])
            stds[i] = np.std(y[m], ddof=0)
            counts[i] = np.sum(m)

    return centers, means, stds, counts


def plot_qco_performance_vs_gain(comp_data, save_dir, prefix="qco", x_mode="relative"):
    rows = comp_data["rows"]

    if x_mode == "relative":
        x = np.array([r["delta_Jphi_rel"] for r in rows], dtype=float)
        xlabel = r"Relative phase-information gain $\Delta J_{\phi}$"
        title = "QCO performance versus relative phase-information gain"
        out_base = f"{prefix}_performance_vs_deltaJphi_rel"
    else:
        x = np.exp(np.array([r["delta_Jphi_log"] for r in rows], dtype=float))
        xlabel = r"Multiplicative phase-information gain $\exp(\Delta J_{\phi}^{\log})$"
        title = "QCO performance versus multiplicative phase-information gain"
        out_base = f"{prefix}_performance_vs_deltaJphi_exp"

    y = np.array([r["delta_rmse"] for r in rows], dtype=float)

    x_for_plot = np.maximum(x, 1e-3) if x_mode == "relative" else x
    stats = binned_statistics_logx(x_for_plot, y, n_bins=18)

    fig, ax = plt.subplots(figsize=(6.7, 5.0))

    ax.scatter(
        x_for_plot, y,
        s=10,
        alpha=0.22,
        color="0.65",
        edgecolors="none",
        label="Simulation points"
    )

    if stats is not None:
        centers, means, stds, counts = stats
        valid = np.isfinite(means)
        ax.plot(
            centers[valid], means[valid],
            color="black",
            lw=2.2,
            label="Binned mean"
        )
        ax.fill_between(
            centers[valid],
            means[valid] - stds[valid],
            means[valid] + stds[valid],
            color="black",
            alpha=0.10,
            linewidth=0.0
        )

    ax.axhline(0.0, ls="--", lw=1.0, color="0.35")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Relative RMSE improvement $\Delta \mathrm{RMSE}$")
    ax.set_title(title)
    ax.legend(frameon=True, loc="upper left")

    finalize_figure(
        fig,
        os.path.join(save_dir, f"{out_base}.png"),
        os.path.join(save_dir, f"{out_base}.pdf")
    )


# =========================================================
# Plot 4: trade-off info gain vs output-noise penalty
# =========================================================
def plot_tradeoff_scatter(comp_data, save_dir, prefix="qco", max_points=1800):
    rows = comp_data["rows"]
    x = np.array([r["delta_Jphi_log"] for r in rows], dtype=float)
    y = np.array([r["delta_Qout"] for r in rows], dtype=float)
    c = np.array([r["delta_rmse"] for r in rows], dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    x = x[mask]
    y = y[mask]
    c = c[mask]

    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
        x_plot = x[idx]
        y_plot = y[idx]
        c_plot = c[idx]
    else:
        x_plot, y_plot, c_plot = x, y, c

    vmin, vmax = robust_sym_limits(c_plot, q=0.99, min_abs=0.03)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    # Main plot
    fig, ax = plt.subplots(figsize=(6.6, 4.9))

    sc = ax.scatter(
        x_plot, y_plot,
        c=c_plot,
        cmap="viridis",
        norm=norm,
        s=14,
        alpha=0.72,
        edgecolors="none",
        rasterized=True
    )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label(r"$\Delta \mathrm{RMSE}$", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    ax.axhline(0.0, ls="--", lw=1.0, color="0.45")
    ax.axvline(0.0, ls="--", lw=1.0, color="0.45")

    ax.set_xlabel(r"Directional information gain $\Delta J_{\phi}^{\log}$")
    ax.set_ylabel(r"Output-noise penalty $\Delta Q_{\mathrm{out}}$")
    ax.set_title("Information gain versus noise penalty")

    ax.grid(True, which="major", alpha=0.18)
    ax.grid(False, which="minor")

    finalize_figure(
        fig,
        os.path.join(save_dir, f"{prefix}_tradeoff_info_vs_noise.png"),
        os.path.join(save_dir, f"{prefix}_tradeoff_info_vs_noise.pdf")
    )

    # Zoom
    y_cap = float(np.quantile(y_plot, 0.88))
    y_cap = max(y_cap, 3.0)

    fig, ax = plt.subplots(figsize=(6.6, 4.9))

    sc = ax.scatter(
        x_plot, y_plot,
        c=c_plot,
        cmap="viridis",
        norm=norm,
        s=14,
        alpha=0.72,
        edgecolors="none",
        rasterized=True
    )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label(r"$\Delta \mathrm{RMSE}$", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    ax.axhline(0.0, ls="--", lw=1.0, color="0.45")
    ax.axvline(0.0, ls="--", lw=1.0, color="0.45")

    ax.set_ylim(min(-0.15, np.min(y_plot) * 1.02), y_cap)
    ax.set_xlabel(r"Directional information gain $\Delta J_{\phi}^{\log}$")
    ax.set_ylabel(r"Output-noise penalty $\Delta Q_{\mathrm{out}}$")
    ax.set_title("Trade-off in the low-penalty regime")

    ax.grid(True, which="major", alpha=0.18)
    ax.grid(False, which="minor")

    finalize_figure(
        fig,
        os.path.join(save_dir, f"{prefix}_tradeoff_info_vs_noise_zoom.png"),
        os.path.join(save_dir, f"{prefix}_tradeoff_info_vs_noise_zoom.pdf")
    )


# =========================================================
# Plot 5: QCO advantage heatmap
# =========================================================
def plot_qco_maps(comp_data, save_dir, prefix="qco"):
    theta_vals = comp_data["theta_vals"]
    lambda_vals = comp_data["lambda_vals"]
    TH, LA = np.meshgrid(theta_vals, lambda_vals)

    save_heatmap(
        TH, LA, comp_data["delta_rmse_map"],
        xlabel=r"Homodyne angle $\theta$",
        ylabel=r"OU rate $\lambda$",
        clabel=r"Relative RMSE improvement $\Delta \mathrm{RMSE}$",
        title=r"QCO advantage map across the $(\theta,\lambda)$ plane",
        out_base=os.path.join(save_dir, f"{prefix}_advantage_map"),
        center_zero=True,
    )

    save_heatmap(
        TH, LA, comp_data["delta_jphi_log_map"],
        xlabel=r"Homodyne angle $\theta$",
        ylabel=r"OU rate $\lambda$",
        clabel=r"Directional information gain $\Delta J_{\phi}^{\log}$",
        title="Directional phase-information gain induced by the QCO",
        out_base=os.path.join(save_dir, f"{prefix}_deltaJphi_log_map"),
        center_zero=False,
    )

    save_heatmap(
        TH, LA, comp_data["delta_qout_map"],
        xlabel=r"Homodyne angle $\theta$",
        ylabel=r"OU rate $\lambda$",
        clabel=r"Output-noise penalty $\Delta Q_{\mathrm{out}}$",
        title="Output-noise penalty induced by the QCO",
        out_base=os.path.join(save_dir, f"{prefix}_deltaQout_map"),
        center_zero=False,
    )


# =========================================================
# Reproducibility report
# =========================================================
def build_reproducibility_report(cfg: SimulationConfig, theta_vals, lambda_vals):
    n_steps = int(cfg.T / cfg.dt)
    burn_steps = int(cfg.burn_fraction * n_steps)
    N_gramian = gramian_horizon(
        gamma=cfg.gamma,
        dt=cfg.dt,
        n_decay=cfg.gramian_n_decay,
        floor=cfg.gramian_floor,
        cap=cfg.gramian_cap,
    )

    return {
        "purpose": "Reproducibility metadata for plant/QCO Monte Carlo simulations",
        "mode": cfg.mode,
        "global_seed": cfg.global_seed,
        "n_mc": cfg.n_mc,
        "dt": cfg.dt,
        "T": cfg.T,
        "n_steps": n_steps,
        "burn_fraction": cfg.burn_fraction,
        "burn_steps": burn_steps,
        "Jphi_gramian_horizon_proxy_N": N_gramian,
        "gramian_rule": {
            "n_decay": cfg.gramian_n_decay,
            "floor": cfg.gramian_floor,
            "cap": cfg.gramian_cap,
            "formula": "N = ceil(n_decay / (gamma*dt)), clipped to [floor, cap]",
        },
        "theta_grid": {
            "min": float(theta_vals[0]),
            "max": float(theta_vals[-1]),
            "count": int(len(theta_vals)),
            "values": [float(v) for v in theta_vals],
        },
        "lambda_grid": {
            "min": float(lambda_vals[0]),
            "max": float(lambda_vals[-1]),
            "count": int(len(lambda_vals)),
            "values": [float(v) for v in lambda_vals],
        },
        "noise_generation": {
            "process_noise": "Gaussian, generated via Cholesky factor of Qd",
            "measurement_noise": "Gaussian, std = meas_std",
            "random_generator": "numpy.random.default_rng",
            "seed_policy": "stable SHA256-based seed derived from (model, theta, lambda, mc_index, global_seed)",
        },
        "plant_parameters": {
            "gamma": cfg.gamma,
            "omega": cfg.omega,
            "g_phi": cfg.g_phi,
            "kappa": cfg.kappa,
            "sigma_q": cfg.sigma_q,
            "sigma_p": cfg.sigma_p,
            "q_phi": cfg.q_phi,
            "meas_std": cfg.meas_std,
        },
        "qco_parameters": {
            "gamma_s": cfg.gamma_s,
            "omega_s": cfg.omega_s,
            "gamma_o": cfg.gamma_o,
            "omega_o": cfg.omega_o,
            "sigma_qs": cfg.sigma_qs,
            "sigma_ps": cfg.sigma_ps,
            "sigma_qo": cfg.sigma_qo,
            "sigma_po": cfg.sigma_po,
            "k_so": cfg.k_so,
            "k_os": cfg.k_os,
            "c_qs": cfg.c_qs,
            "c_ps": cfg.c_ps,
            "c_qo": cfg.c_qo,
            "c_po": cfg.c_po,
        },
    }


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    cfg = SimulationConfig()

    if cfg.mode == "FAST":
        cfg.n_mc = 2
        cfg.T = 25.0
        cfg.dt = 0.025
        cfg.n_theta = 28
        cfg.n_lambda = 24
        cfg.noncollapse_max_points = 900
    elif cfg.mode == "PAPER_HEAVY":
        cfg.n_mc = 5
        cfg.T = 40.0
        cfg.dt = 0.02
        cfg.n_theta = 48
        cfg.n_lambda = 36
        cfg.noncollapse_max_points = 1200
    elif cfg.mode == "FINAL_HEAVY":
        cfg.n_mc = 6
        cfg.T = 60.0
        cfg.dt = 0.015
        cfg.n_theta = 64
        cfg.n_lambda = 48
        cfg.noncollapse_max_points = 1600

    ensure_dir(cfg.save_dir)

    theta_vals = np.linspace(cfg.theta_min, cfg.theta_max, cfg.n_theta)
    lambda_vals = np.linspace(cfg.lambda_min, cfg.lambda_max, cfg.n_lambda)

    print("=================================================")
    print("Running paper-ready plant/QCO study")
    print(f"Mode                 : {cfg.mode}")
    print(f"Grid                 : {len(theta_vals)} x {len(lambda_vals)} = {len(theta_vals)*len(lambda_vals)} points")
    print(f"Monte Carlo runs     : {cfg.n_mc}")
    print(f"T, dt                : {cfg.T}, {cfg.dt}")
    print(f"Burn-in fraction     : {cfg.burn_fraction}")
    print(f"Global seed          : {cfg.global_seed}")
    print(f"Hexbin non-collapse  : {cfg.noncollapse_use_hexbin}")
    print("=================================================")

    if cfg.save_config_json:
        export_json(asdict(cfg), os.path.join(cfg.save_dir, "simulation_config.json"))
        report = build_reproducibility_report(cfg, theta_vals, lambda_vals)
        export_json(report, os.path.join(cfg.save_dir, "reproducibility_report.json"))

    # Plant grid
    plant_data = collect_plant_grid(theta_vals, lambda_vals, cfg)
    if cfg.save_raw_csv:
        export_csv_dicts(
            plant_data["rows"],
            os.path.join(cfg.save_dir, "plant_grid_results.csv")
        )

    # Plant plots
    if cfg.do_plant_families:
        plot_families_fixed_lambda(plant_data, cfg.save_dir, prefix="plant")

    if cfg.do_plant_global_noncollapse:
        plot_global_noncollapse(
            plant_data,
            cfg.save_dir,
            prefix="plant",
            max_points=cfg.noncollapse_max_points,
            use_hexbin=cfg.noncollapse_use_hexbin
        )

    # QCO comparisons
    if cfg.do_qco_map or cfg.do_qco_performance_scatter or cfg.do_tradeoff_scatter:
        comp_data = collect_qco_comparison_grid(plant_data, cfg)

        if cfg.save_raw_csv:
            export_csv_dicts(
                comp_data["rows"],
                os.path.join(cfg.save_dir, "qco_vs_plant_results.csv")
            )

        if cfg.do_qco_map:
            plot_qco_maps(comp_data, cfg.save_dir, prefix="qco")

        if cfg.do_qco_performance_scatter:
            plot_qco_performance_vs_gain(
                comp_data, cfg.save_dir, prefix="qco", x_mode="relative"
            )

        if cfg.do_tradeoff_scatter:
            plot_tradeoff_scatter(comp_data, cfg.save_dir, prefix="qco")

    print("\nDone. Outputs written to:", cfg.save_dir)