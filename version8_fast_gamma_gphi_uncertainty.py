
"""
Optimized paper-oriented analysis script.

Main upgrades vs the original version:
1) Much faster Monte Carlo using batched Kalman simulation.
2) Plant baseline is cached once and reused for QCO comparisons.
3) Strategic gamma and g_phi sweeps:
   - no full theta-lambda rerun for every parameter value by default
   - reduced sweeps at selected informative angles
4) Uncertainty bars:
   - MC mean/std/SEM
   - optional bootstrap CI for fitted law RMSE ≈ a/sqrt(J_phi) + b
5) Explicit fit of RMSE = a / sqrt(J_phi) + b, globally and by family.

Expected project imports:
- core.model
- core.qco_model
"""

import os
import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from joblib import Parallel, delayed

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

EPS = 1e-12
DEFAULT_C_PS = 1.0
DEFAULT_C_PO = 0.4


# =========================================================
# Config
# =========================================================
@dataclass
class RunConfig:
    save_dir: str = "plots_fast_paper"
    mode: str = "PAPER"      # DEBUG / PAPER / FINAL
    n_jobs: int = max(1, (os.cpu_count() or 2) - 1)

    # Save
    save_png: bool = True
    save_pdf: bool = True
    save_npz_data: bool = True
    save_json_data: bool = True

    # Simulation
    dt: float = 0.01
    T: float = 25.0
    n_mc: int = 12
    burn_frac: float = 0.2

    # Plant nominal
    gamma: float = 1.0
    omega: float = 0.0
    g_phi: float = 2.0
    kappa: float = 1.0
    q_phi: float = 0.01
    meas_std: float = 0.03
    sigma_q: float = 0.02
    sigma_p: float = 0.02

    # QCO nominal
    gamma_s: float = 1.0
    omega_s: float = 0.0
    gamma_o: float = 1.2
    omega_o: float = 0.4
    k_so: float = 0.8
    k_os: float = 0.8
    sigma_qs: float = 0.02
    sigma_ps: float = 0.02
    sigma_qo: float = 0.02
    sigma_po: float = 0.02
    c_qs: float = 0.0
    c_ps: float = DEFAULT_C_PS
    c_qo: float = 0.0
    c_po: float = DEFAULT_C_PO

    # Main grids
    theta_vals: np.ndarray = field(default_factory=lambda: np.linspace(0.05, np.pi - 0.05, 21))
    lambda_vals: np.ndarray = field(default_factory=lambda: np.linspace(0.02, 1.5, 18))

    # Strategic sweeps
    gamma_vals: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.7, 1.0, 1.5, 2.0, 3.0]))
    gphi_vals: np.ndarray = field(default_factory=lambda: np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0]))

    # Plot switches
    do_nominal_theta_lambda: bool = True
    do_qco_theta_lambda: bool = True
    do_gamma_strategic: bool = True
    do_gphi_strategic: bool = True
    do_fit_uncertainty: bool = True

    # Fit
    n_boot: int = 300
    family_fit_min_points: int = 8

    def __post_init__(self):
        mode = self.mode.upper()
        if mode == "DEBUG":
            self.T = 12.0
            self.n_mc = 4
            self.theta_vals = np.linspace(0.05, np.pi - 0.05, 11)
            self.lambda_vals = np.linspace(0.02, 1.5, 9)
            self.gamma_vals = np.array([0.7, 1.0, 1.5])
            self.gphi_vals = np.array([1.0, 2.0, 3.0])
            self.n_boot = 80
        elif mode == "FINAL":
            self.T = 40.0
            self.n_mc = 20
            self.theta_vals = np.linspace(0.05, np.pi - 0.05, 35)
            self.lambda_vals = np.linspace(0.02, 1.5, 28)
            self.gamma_vals = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
            self.gphi_vals = np.array([0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
            self.n_boot = 600


# =========================================================
# Small utilities
# =========================================================
def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def stable_seed(*vals) -> int:
    key = "_".join([f"{float(v):.12e}" for v in vals])
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def save_npz(save_dir: str, fname: str, **arrays):
    ensure_dir(save_dir)
    np.savez(Path(save_dir) / f"{fname}.npz", **arrays)


def save_json(save_dir: str, fname: str, payload: dict):
    ensure_dir(save_dir)
    with open(Path(save_dir) / f"{fname}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_figure(fig, save_dir: str, fname: str, save_png=True, save_pdf=True, dpi=250):
    ensure_dir(save_dir)
    if save_png:
        fig.savefig(Path(save_dir) / f"{fname}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    if save_pdf:
        fig.savefig(Path(save_dir) / f"{fname}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def setup_axes(ax, xlabel=None, ylabel=None, title=None, grid=True):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.25, linewidth=0.6)


def gramian_horizon(gamma: float, dt: float, n_decay: float = 5.0, floor: int = 20, cap: int = 200) -> int:
    steps = int(np.ceil(n_decay / max(gamma * dt, EPS)))
    return max(floor, min(steps, cap))


def safe_inv_sqrt(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / np.sqrt(np.maximum(x, EPS))


def effective_noise_proxy(Qd: np.ndarray) -> float:
    return float(np.trace(Qd))


def mean_std_sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    sem = s / np.sqrt(max(len(x), 1))
    return m, s, sem


def quantile_interval(x, q_low=0.16, q_high=0.84):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    return float(np.quantile(x, q_low)), float(np.quantile(x, q_high))


def robust_signed_limits(Z, q=0.98, eps=1e-12):
    z = np.asarray(Z, dtype=float)
    z = z[np.isfinite(z)]
    if len(z) == 0:
        return -1.0, 1.0
    vmax = max(float(np.quantile(np.abs(z), q)), eps)
    return -vmax, vmax


def robust_positive_limits(Z, q_low=0.02, q_high=0.98, eps=1e-12):
    z = np.asarray(Z, dtype=float)
    z = z[np.isfinite(z)]
    if len(z) == 0:
        return eps, 1.0
    vmin = max(float(np.quantile(z, q_low)), eps)
    vmax = max(float(np.quantile(z, q_high)), vmin + eps)
    return vmin, vmax


def run_parallel(tasks, fn, n_jobs=1, desc=""):
    if desc:
        print(f"[run] {desc} | tasks={len(tasks)} | n_jobs={n_jobs}", flush=True)
    if n_jobs == 1:
        return [fn(*t) for t in tasks]
    return Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(fn)(*t) for t in tasks
    )


# =========================================================
# Observability
# =========================================================
def discrete_observability_gramian(F: np.ndarray, H: np.ndarray, horizon: int) -> np.ndarray:
    n = F.shape[0]
    Wo = np.zeros((n, n), dtype=float)
    Fk = np.eye(n)
    for _ in range(horizon):
        HFk = H @ Fk
        Wo += HFk.T @ HFk
        Fk = F @ Fk
    return Wo


def phase_observability_metric(F: np.ndarray, H: np.ndarray, phase_index: int, horizon: int) -> float:
    e = np.zeros(F.shape[0], dtype=float)
    e[phase_index] = 1.0
    Fk = np.eye(F.shape[0])
    J = 0.0
    for _ in range(horizon):
        out = H @ (Fk @ e)
        J += float(np.dot(out.ravel(), out.ravel()))
        Fk = F @ Fk
    return float(J)


def gramian_metrics(F, H, horizon, phase_index):
    Wo = discrete_observability_gramian(F, H, horizon)
    eigvals = np.maximum(np.linalg.eigvalsh(Wo), 0.0)
    lam_min = float(np.min(eigvals))
    lam_max = float(np.max(eigvals))
    cond = np.inf if lam_min < 1e-14 else float(lam_max / lam_min)
    J_phi = phase_observability_metric(F, H, phase_index, horizon)
    return {
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "cond": cond,
        "trace": float(np.trace(Wo)),
        "J_phi": J_phi,
    }


# =========================================================
# Fast batched Kalman
# =========================================================
def precompute_kalman_gains(F, H, Qd, R, P0, n_steps):
    n = F.shape[0]
    H = H.reshape(1, n)
    P_post = P0.copy()
    Ks = np.zeros((n_steps, n), dtype=float)

    eye_n = np.eye(n)
    for k in range(n_steps):
        P_pred = F @ P_post @ F.T + Qd
        S = float((H @ P_pred @ H.T)[0, 0] + R[0, 0])
        K = (P_pred @ H.T / max(S, EPS)).reshape(-1)
        P_post = (eye_n - np.outer(K, H.reshape(-1))) @ P_pred
        P_post = 0.5 * (P_post + P_post.T)
        Ks[k] = K
    return Ks, P_post


def batched_mc_phase_stats(F, H, Qd, R, x_true0, x_hat0, P0, T, dt, phase_index, n_mc=8, burn_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    n_steps = int(np.round(T / dt))
    burn = int(burn_frac * n_steps)
    n = F.shape[0]
    Hvec = H.reshape(-1)

    Ks, Pss = precompute_kalman_gains(F, H, Qd, R, P0, n_steps)

    try:
        Lq = np.linalg.cholesky(Qd + 1e-15 * np.eye(n))
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(0.5 * (Qd + Qd.T))
        evals = np.clip(evals, 0.0, None)
        Lq = evecs @ np.diag(np.sqrt(evals))

    meas_std = np.sqrt(max(R[0, 0], EPS))
    x_true = np.tile(x_true0.reshape(1, n), (n_mc, 1))
    x_est = np.tile(x_hat0.reshape(1, n), (n_mc, 1))
    err_sq_sum = np.zeros(n_mc, dtype=float)
    counts = np.zeros(n_mc, dtype=int)

    for k in range(n_steps):
        w = rng.standard_normal((n_mc, n)) @ Lq.T
        v = rng.normal(0.0, meas_std, size=n_mc)

        x_true = x_true @ F.T + w
        y = x_true @ Hvec + v

        x_pred = x_est @ F.T
        innov = y - (x_pred @ Hvec)
        x_est = x_pred + innov[:, None] * Ks[k][None, :]

        if k >= burn:
            diff = x_true[:, phase_index] - x_est[:, phase_index]
            err_sq_sum += diff ** 2
            counts += 1

    rmse_per_run = np.sqrt(err_sq_sum / np.maximum(counts, 1))
    mean_rmse, std_rmse, sem_rmse = mean_std_sem(rmse_per_run)
    low_rmse, high_rmse = quantile_interval(rmse_per_run)
    ss_phi_std = float(np.sqrt(max(Pss[phase_index, phase_index], EPS)))

    return {
        "rmse_mean": mean_rmse,
        "rmse_std": std_rmse,
        "rmse_sem": sem_rmse,
        "rmse_q16": low_rmse,
        "rmse_q84": high_rmse,
        "rmse_per_run": rmse_per_run,
        "ss_phi_std": ss_phi_std,
    }


# =========================================================
# System builders
# =========================================================
def build_plant_system(theta, lam, cfg: RunConfig, gamma=None, g_phi=None):
    gamma = cfg.gamma if gamma is None else gamma
    g_phi = cfg.g_phi if g_phi is None else g_phi
    A = build_Aa(gamma=gamma, omega=cfg.omega, g_phi=g_phi, lam=lam)
    H = build_Ca(kappa=cfg.kappa, theta=theta)
    Qc = build_continuous_process_covariance(
        sigma_q=cfg.sigma_q,
        sigma_p=cfg.sigma_p,
        q_phi=cfg.q_phi,
    )
    F, Qd = discretize_plant(A, Qc, cfg.dt)
    R = np.array([[cfg.meas_std ** 2]], dtype=float)
    return F, H, Qd, R


def build_qco_system(theta, lam, cfg: RunConfig, gamma_s=None, g_phi=None, k_so=None, k_os=None):
    gamma_s = cfg.gamma_s if gamma_s is None else gamma_s
    g_phi = cfg.g_phi if g_phi is None else g_phi
    k_so = cfg.k_so if k_so is None else k_so
    k_os = cfg.k_os if k_os is None else k_os

    A = build_qco_augmented_A(
        gamma_s=gamma_s,
        omega_s=cfg.omega_s,
        gamma_o=cfg.gamma_o,
        omega_o=cfg.omega_o,
        g_phi=g_phi,
        lam=lam,
        k_so=k_so,
        k_os=k_os,
    )
    H = build_qco_measurement(
        c_qs=cfg.c_qs,
        c_ps=cfg.c_ps,
        c_qo=cfg.c_qo,
        c_po=cfg.c_po,
    )
    Qc = build_qco_process_covariance(
        sigma_qs=cfg.sigma_qs,
        sigma_ps=cfg.sigma_ps,
        sigma_qo=cfg.sigma_qo,
        sigma_po=cfg.sigma_po,
        q_phi=cfg.q_phi,
    )
    F, Qd = discretize_qco(A, Qc, cfg.dt)
    R = np.array([[cfg.meas_std ** 2]], dtype=float)
    return F, H, Qd, R


# =========================================================
# Metrics
# =========================================================
def plant_metrics(theta, lam, cfg: RunConfig, gamma=None, g_phi=None):
    gamma = cfg.gamma if gamma is None else gamma
    g_phi = cfg.g_phi if g_phi is None else g_phi
    F, H, Qd, R = build_plant_system(theta, lam, cfg, gamma=gamma, g_phi=g_phi)

    x_true0 = np.array([0.0, 0.0, 0.5], dtype=float)
    x_hat0 = np.zeros(3, dtype=float)
    P0 = 10.0 * np.eye(3)
    seed = stable_seed(theta, lam, gamma, g_phi, cfg.q_phi, cfg.meas_std)

    stats = batched_mc_phase_stats(
        F, H, Qd, R, x_true0, x_hat0, P0,
        T=cfg.T, dt=cfg.dt, phase_index=2,
        n_mc=cfg.n_mc, burn_frac=cfg.burn_frac, seed=seed
    )
    gram = gramian_metrics(F, H, gramian_horizon(gamma, cfg.dt), phase_index=2)

    return {
        "F": F, "H": H, "Qd": Qd, "R": R,
        "rmse_phi": stats["rmse_mean"],
        "rmse_std": stats["rmse_std"],
        "rmse_sem": stats["rmse_sem"],
        "rmse_q16": stats["rmse_q16"],
        "rmse_q84": stats["rmse_q84"],
        "rmse_per_run": stats["rmse_per_run"],
        "ss_phi_std": stats["ss_phi_std"],
        "noise_proxy": effective_noise_proxy(Qd),
        **gram,
    }


def qco_metrics(theta, lam, cfg: RunConfig, gamma_s=None, g_phi=None, k_so=None, k_os=None):
    gamma_s = cfg.gamma_s if gamma_s is None else gamma_s
    g_phi = cfg.g_phi if g_phi is None else g_phi
    k_so = cfg.k_so if k_so is None else k_so
    k_os = cfg.k_os if k_os is None else k_os
    F, H, Qd, R = build_qco_system(theta, lam, cfg, gamma_s=gamma_s, g_phi=g_phi, k_so=k_so, k_os=k_os)

    x_true0 = np.array([0.0, 0.0, 0.0, 0.0, 0.5], dtype=float)
    x_hat0 = np.zeros(5, dtype=float)
    P0 = 10.0 * np.eye(5)
    seed = stable_seed(theta, lam, gamma_s, g_phi, cfg.q_phi, cfg.meas_std, k_so, k_os)

    stats = batched_mc_phase_stats(
        F, H, Qd, R, x_true0, x_hat0, P0,
        T=cfg.T, dt=cfg.dt, phase_index=4,
        n_mc=cfg.n_mc, burn_frac=cfg.burn_frac, seed=seed
    )
    gram = gramian_metrics(F, H, gramian_horizon(gamma_s, cfg.dt), phase_index=4)

    return {
        "F": F, "H": H, "Qd": Qd, "R": R,
        "rmse_phi": stats["rmse_mean"],
        "rmse_std": stats["rmse_std"],
        "rmse_sem": stats["rmse_sem"],
        "rmse_q16": stats["rmse_q16"],
        "rmse_q84": stats["rmse_q84"],
        "rmse_per_run": stats["rmse_per_run"],
        "ss_phi_std": stats["ss_phi_std"],
        "noise_proxy": effective_noise_proxy(Qd),
        **gram,
    }


# =========================================================
# Dataset collection
# =========================================================
def collect_plant_theta_lambda(cfg: RunConfig, gamma=None, g_phi=None, dataset_name="plant_nominal"):
    theta_vals = np.asarray(cfg.theta_vals, dtype=float)
    lambda_vals = np.asarray(cfg.lambda_vals, dtype=float)
    tasks = [(i, j, theta_vals[j], lambda_vals[i]) for i in range(len(lambda_vals)) for j in range(len(theta_vals))]

    def _one(i, j, theta, lam):
        met = plant_metrics(theta, lam, cfg, gamma=gamma, g_phi=g_phi)
        return i, j, met

    results = run_parallel(tasks, _one, n_jobs=cfg.n_jobs, desc=f"collect_plant_theta_lambda[{dataset_name}]")

    shape = (len(lambda_vals), len(theta_vals))
    rmse_map = np.zeros(shape)
    rmse_sem_map = np.zeros(shape)
    ss_phi_map = np.zeros(shape)
    jphi_map = np.zeros(shape)
    cond_map = np.zeros(shape)
    lambda_min_map = np.zeros(shape)
    rows = []
    cache = {}

    for i, j, met in results:
        theta = theta_vals[j]
        lam = lambda_vals[i]
        rmse_map[i, j] = met["rmse_phi"]
        rmse_sem_map[i, j] = met["rmse_sem"]
        ss_phi_map[i, j] = met["ss_phi_std"]
        jphi_map[i, j] = met["J_phi"]
        cond_map[i, j] = met["cond"]
        lambda_min_map[i, j] = met["lambda_min"]
        rows.append({
            "theta": theta,
            "lambda": lam,
            "rmse_phi": met["rmse_phi"],
            "rmse_sem": met["rmse_sem"],
            "ss_phi_std": met["ss_phi_std"],
            "J_phi": met["J_phi"],
            "inv_sqrt_Jphi": 1.0 / np.sqrt(max(met["J_phi"], EPS)),
            "cond": met["cond"],
            "lambda_min": met["lambda_min"],
        })
        cache[(i, j)] = met

    if cfg.save_npz_data:
        save_npz(
            cfg.save_dir, dataset_name,
            theta_vals=theta_vals,
            lambda_vals=lambda_vals,
            rmse_map=rmse_map,
            rmse_sem_map=rmse_sem_map,
            ss_phi_map=ss_phi_map,
            jphi_map=jphi_map,
            cond_map=cond_map,
            lambda_min_map=lambda_min_map,
        )

    return {
        "theta_vals": theta_vals,
        "lambda_vals": lambda_vals,
        "rmse_map": rmse_map,
        "rmse_sem_map": rmse_sem_map,
        "ss_phi_map": ss_phi_map,
        "jphi_map": jphi_map,
        "cond_map": cond_map,
        "lambda_min_map": lambda_min_map,
        "rows": rows,
        "cache": cache,
    }


def collect_qco_vs_cached_plant(cfg: RunConfig, plant_data: dict, gamma_s=None, g_phi=None, dataset_name="qco_nominal"):
    theta_vals = plant_data["theta_vals"]
    lambda_vals = plant_data["lambda_vals"]
    tasks = [(i, j, theta_vals[j], lambda_vals[i]) for i in range(len(lambda_vals)) for j in range(len(theta_vals))]

    def _one(i, j, theta, lam):
        plant = plant_data["cache"][(i, j)]
        qco = qco_metrics(theta, lam, cfg, gamma_s=gamma_s, g_phi=g_phi)
        delta_rmse = (plant["rmse_phi"] - qco["rmse_phi"]) / max(abs(plant["rmse_phi"]), EPS)
        delta_rmse_sem = np.sqrt(plant["rmse_sem"]**2 + qco["rmse_sem"]**2) / max(abs(plant["rmse_phi"]), EPS)
        delta_J_rel = (qco["J_phi"] - plant["J_phi"]) / max(abs(plant["J_phi"]), EPS)
        delta_J_log = np.log(qco["J_phi"] + EPS) - np.log(plant["J_phi"] + EPS)
        delta_Q = (qco["noise_proxy"] - plant["noise_proxy"]) / max(abs(plant["noise_proxy"]), EPS)
        return i, j, {
            "plant": plant,
            "qco": qco,
            "delta_rmse": delta_rmse,
            "delta_rmse_sem": delta_rmse_sem,
            "delta_Jphi": delta_J_rel,
            "delta_Jphi_log": delta_J_log,
            "delta_Q": delta_Q,
        }

    results = run_parallel(tasks, _one, n_jobs=cfg.n_jobs, desc=f"collect_qco_vs_cached_plant[{dataset_name}]")

    shape = (len(lambda_vals), len(theta_vals))
    out = {
        "theta_vals": theta_vals,
        "lambda_vals": lambda_vals,
        "delta_rmse": np.zeros(shape),
        "delta_rmse_sem": np.zeros(shape),
        "delta_Jphi": np.zeros(shape),
        "delta_Jphi_log": np.zeros(shape),
        "delta_Q": np.zeros(shape),
        "Jphi_plant": np.zeros(shape),
        "Jphi_qco": np.zeros(shape),
        "rmse_plant": np.zeros(shape),
        "rmse_qco": np.zeros(shape),
    }

    for i, j, comp in results:
        out["delta_rmse"][i, j] = comp["delta_rmse"]
        out["delta_rmse_sem"][i, j] = comp["delta_rmse_sem"]
        out["delta_Jphi"][i, j] = comp["delta_Jphi"]
        out["delta_Jphi_log"][i, j] = comp["delta_Jphi_log"]
        out["delta_Q"][i, j] = comp["delta_Q"]
        out["Jphi_plant"][i, j] = comp["plant"]["J_phi"]
        out["Jphi_qco"][i, j] = comp["qco"]["J_phi"]
        out["rmse_plant"][i, j] = comp["plant"]["rmse_phi"]
        out["rmse_qco"][i, j] = comp["qco"]["rmse_phi"]

    if cfg.save_npz_data:
        save_npz(cfg.save_dir, dataset_name, **out)
    return out


# =========================================================
# Fits RMSE = a / sqrt(Jphi) + b
# =========================================================
def fit_a_invrootJ_plus_b(Jphi, rmse):
    x = safe_inv_sqrt(Jphi).ravel()
    y = np.asarray(rmse, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if len(x) < 3 or np.std(x) < EPS:
        return {"a": np.nan, "b": np.nan, "r2": np.nan, "rmse_fit": np.nan, "n": len(x)}
    X = np.column_stack([x, np.ones(len(x))])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot < EPS else 1.0 - ss_res / ss_tot
    fit_rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return {"a": float(beta[0]), "b": float(beta[1]), "r2": r2, "rmse_fit": fit_rmse, "n": len(x)}


def bootstrap_fit_a_invrootJ_plus_b(Jphi, rmse, n_boot=200, seed=0):
    x = safe_inv_sqrt(Jphi).ravel()
    y = np.asarray(rmse, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    n = len(x)
    if n < 5:
        return {"a_ci": [np.nan, np.nan], "b_ci": [np.nan, np.nan], "r2_ci": [np.nan, np.nan]}
    rng = np.random.default_rng(seed)
    a_list, b_list, r2_list = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        fit = fit_a_invrootJ_plus_b(1.0 / np.maximum(x[idx], EPS)**2, y[idx])
        a_list.append(fit["a"])
        b_list.append(fit["b"])
        r2_list.append(fit["r2"])
    a_ci = [float(np.quantile(a_list, 0.16)), float(np.quantile(a_list, 0.84))]
    b_ci = [float(np.quantile(b_list, 0.16)), float(np.quantile(b_list, 0.84))]
    r2_ci = [float(np.quantile(r2_list, 0.16)), float(np.quantile(r2_list, 0.84))]
    return {"a_ci": a_ci, "b_ci": b_ci, "r2_ci": r2_ci}


def analyze_fit_global_and_families(plant_data: dict, cfg: RunConfig, prefix="plant_nominal"):
    Jphi = plant_data["jphi_map"]
    rmse = plant_data["rmse_map"]
    theta_vals = plant_data["theta_vals"]

    global_fit = fit_a_invrootJ_plus_b(Jphi, rmse)
    global_ci = bootstrap_fit_a_invrootJ_plus_b(Jphi, rmse, n_boot=cfg.n_boot, seed=stable_seed(1.234, 5.678))

    family_fits = []
    for j, theta in enumerate(theta_vals):
        fit = fit_a_invrootJ_plus_b(Jphi[:, j], rmse[:, j])
        if fit["n"] >= cfg.family_fit_min_points:
            fit["theta"] = float(theta)
            family_fits.append(fit)

    out = {"global_fit": global_fit, "global_ci": global_ci, "family_fits": family_fits}
    if cfg.save_json_data:
        save_json(cfg.save_dir, f"{prefix}_fit_invrootJ_plus_b", out)
    return out


# =========================================================
# Strategic sweeps in gamma and g_phi
# =========================================================
def extract_informative_thetas(plant_data: dict) -> Dict[str, float]:
    theta_vals = plant_data["theta_vals"]
    lambda_vals = plant_data["lambda_vals"]
    rmse_map = plant_data["rmse_map"]
    jphi_map = plant_data["jphi_map"]

    idx_mid = len(lambda_vals) // 2
    theta_rmse_mid = float(theta_vals[int(np.argmin(rmse_map[idx_mid, :]))])
    theta_jphi_mid = float(theta_vals[int(np.argmax(jphi_map[idx_mid, :]))])
    theta_pi2 = float(np.pi / 2)

    vals = np.array([theta_rmse_mid, theta_jphi_mid, theta_pi2], dtype=float)
    vals = np.unique(np.round(vals, 6))
    return {
        "theta_rmse_mid": theta_rmse_mid,
        "theta_jphi_mid": theta_jphi_mid,
        "theta_pi2": theta_pi2,
        "theta_list": vals.tolist(),
    }


def strategic_parameter_sweep(cfg: RunConfig, param_name: str, param_vals: np.ndarray, theta_list: List[float], lambda_vals: np.ndarray, prefix="gamma"):
    assert param_name in {"gamma", "g_phi"}
    rows = []

    tasks = []
    for p in param_vals:
        for theta in theta_list:
            for lam in lambda_vals:
                tasks.append((float(p), float(theta), float(lam)))

    def _one(p, theta, lam):
        if param_name == "gamma":
            plant = plant_metrics(theta, lam, cfg, gamma=p, g_phi=cfg.g_phi)
            qco = qco_metrics(theta, lam, cfg, gamma_s=p, g_phi=cfg.g_phi)
        else:
            plant = plant_metrics(theta, lam, cfg, gamma=cfg.gamma, g_phi=p)
            qco = qco_metrics(theta, lam, cfg, gamma_s=cfg.gamma_s, g_phi=p)

        delta_rmse = (plant["rmse_phi"] - qco["rmse_phi"]) / max(abs(plant["rmse_phi"]), EPS)
        delta_rmse_sem = np.sqrt(plant["rmse_sem"]**2 + qco["rmse_sem"]**2) / max(abs(plant["rmse_phi"]), EPS)
        delta_J = (qco["J_phi"] - plant["J_phi"]) / max(abs(plant["J_phi"]), EPS)
        return {
            "param": p,
            "theta": theta,
            "lambda": lam,
            "rmse_plant": plant["rmse_phi"],
            "rmse_qco": qco["rmse_phi"],
            "delta_rmse": delta_rmse,
            "delta_rmse_sem": delta_rmse_sem,
            "Jphi_plant": plant["J_phi"],
            "Jphi_qco": qco["J_phi"],
            "delta_Jphi": delta_J,
        }

    out_rows = run_parallel(tasks, _one, n_jobs=cfg.n_jobs, desc=f"strategic_{param_name}_sweep")

    # Aggregate across lambdas for each (param, theta)
    for p in param_vals:
        for theta in theta_list:
            sub = [r for r in out_rows if abs(r["param"] - p) < 1e-15 and abs(r["theta"] - theta) < 1e-12]
            d_rmse = np.array([r["delta_rmse"] for r in sub], dtype=float)
            dJ = np.array([r["delta_Jphi"] for r in sub], dtype=float)
            rmse_p = np.array([r["rmse_plant"] for r in sub], dtype=float)
            rmse_q = np.array([r["rmse_qco"] for r in sub], dtype=float)

            m1, s1, sem1 = mean_std_sem(d_rmse)
            lo1, hi1 = quantile_interval(d_rmse)
            m2, s2, sem2 = mean_std_sem(dJ)
            rows.append({
                "param": float(p),
                "theta": float(theta),
                "delta_rmse_mean": m1,
                "delta_rmse_std": s1,
                "delta_rmse_sem": sem1,
                "delta_rmse_q16": lo1,
                "delta_rmse_q84": hi1,
                "delta_Jphi_mean": m2,
                "delta_Jphi_std": s2,
                "delta_Jphi_sem": sem2,
                "rmse_plant_mean": float(np.mean(rmse_p)),
                "rmse_qco_mean": float(np.mean(rmse_q)),
            })

    payload = {"rows": rows, "raw_rows": out_rows, "param_name": param_name}
    if cfg.save_json_data:
        save_json(cfg.save_dir, f"strategic_{prefix}_summary", payload)
    return payload


# =========================================================
# Plotting
# =========================================================
def _save_map(X, Y, Z, xlabel, ylabel, clabel, title, save_dir, fname, signed=True):
    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    Z = np.asarray(Z, dtype=float)
    if signed:
        vmin, vmax = robust_signed_limits(Z)
        im = ax.pcolormesh(X, Y, Z, shading="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax))
    else:
        vmin, vmax = robust_positive_limits(Z)
        im = ax.pcolormesh(X, Y, Z, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(clabel)
    setup_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title, grid=False)
    save_figure(fig, save_dir, fname)


def plot_nominal_maps(plant_data: dict, qco_data: dict, cfg: RunConfig, prefix="nominal"):
    TH, LA = np.meshgrid(plant_data["theta_vals"], plant_data["lambda_vals"])

    _save_map(
        TH, LA, plant_data["rmse_map"],
        r"Homodyne angle $\theta$", r"OU rate $\lambda$",
        r"$\mathrm{RMSE}(\phi)$", "Plant RMSE map",
        cfg.save_dir, f"{prefix}_plant_rmse", signed=False,
    )
    _save_map(
        TH, LA, plant_data["jphi_map"],
        r"Homodyne angle $\theta$", r"OU rate $\lambda$",
        r"$J_\phi(N)$", "Directional phase metric map",
        cfg.save_dir, f"{prefix}_plant_jphi", signed=False,
    )
    _save_map(
        TH, LA, qco_data["delta_rmse"],
        r"Homodyne angle $\theta$", r"OU rate $\lambda$",
        r"$\Delta_{\rm RMSE}$", "QCO advantage map",
        cfg.save_dir, f"{prefix}_qco_delta_rmse", signed=True,
    )


def plot_fit_invrootJ(plant_data: dict, fit_summary: dict, cfg: RunConfig, prefix="nominal"):
    rows = plant_data["rows"]
    J = np.array([r["J_phi"] for r in rows], dtype=float)
    rmse = np.array([r["rmse_phi"] for r in rows], dtype=float)
    lam = np.array([r["lambda"] for r in rows], dtype=float)
    x = safe_inv_sqrt(J)

    fit = fit_summary["global_fit"]
    fig, ax = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)
    sc = ax.scatter(x, rmse, c=lam, s=16, alpha=0.75)
    fig.colorbar(sc, ax=ax, label=r"$\lambda$")
    if np.isfinite(fit["a"]):
        xx = np.linspace(np.min(x), np.max(x), 200)
        yy = fit["a"] * xx + fit["b"]
        ax.plot(xx, yy, linewidth=2.2, label=rf"fit: $a/\sqrt{{J_\phi}}+b$")
        ax.legend()
    setup_axes(ax, xlabel=r"$1/\sqrt{J_\phi(N)}$", ylabel=r"$\mathrm{RMSE}(\phi)$", title="Global fit with affine offset")
    save_figure(fig, cfg.save_dir, f"{prefix}_fit_invrootJ_plus_b", cfg.save_png, cfg.save_pdf)


def plot_strategic_sweep(summary: dict, cfg: RunConfig, param_symbol: str, prefix="gamma"):
    rows = summary["rows"]
    thetas = sorted({round(r["theta"], 12) for r in rows})
    params = sorted({round(r["param"], 12) for r in rows})

    # Delta RMSE with uncertainty
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for th in thetas:
        sub = [r for r in rows if abs(r["theta"] - th) < 1e-12]
        p = np.array([r["param"] for r in sub], dtype=float)
        m = np.array([r["delta_rmse_mean"] for r in sub], dtype=float)
        lo = np.array([r["delta_rmse_q16"] for r in sub], dtype=float)
        hi = np.array([r["delta_rmse_q84"] for r in sub], dtype=float)
        ax.plot(p, m, marker="o", linewidth=2.0, label=rf"$\theta={th:.2f}$")
        ax.fill_between(p, lo, hi, alpha=0.18)
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="k", alpha=0.7)
    setup_axes(ax, xlabel=param_symbol, ylabel=r"mean $\Delta_{\rm RMSE}$ over $\lambda$", title=rf"QCO gain vs {param_symbol}")
    ax.legend(fontsize=8)
    save_figure(fig, cfg.save_dir, f"strategic_{prefix}_delta_rmse", cfg.save_png, cfg.save_pdf)

    # Delta Jphi
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for th in thetas:
        sub = [r for r in rows if abs(r["theta"] - th) < 1e-12]
        p = np.array([r["param"] for r in sub], dtype=float)
        m = np.array([r["delta_Jphi_mean"] for r in sub], dtype=float)
        sem = np.array([r["delta_Jphi_sem"] for r in sub], dtype=float)
        ax.plot(p, m, marker="s", linewidth=2.0, label=rf"$\theta={th:.2f}$")
        ax.fill_between(p, m - sem, m + sem, alpha=0.18)
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="k", alpha=0.7)
    setup_axes(ax, xlabel=param_symbol, ylabel=r"mean $\Delta J_\phi$ over $\lambda$", title=rf"Phase-information gain vs {param_symbol}")
    ax.legend(fontsize=8)
    save_figure(fig, cfg.save_dir, f"strategic_{prefix}_delta_Jphi", cfg.save_png, cfg.save_pdf)


# =========================================================
# Main
# =========================================================
def main():
    cfg = RunConfig()
    ensure_dir(cfg.save_dir)

    print(f"Mode={cfg.mode} | n_jobs={cfg.n_jobs} | n_mc={cfg.n_mc}", flush=True)

    plant_nom = None
    qco_nom = None
    fit_nom = None
    informative = None

    if cfg.do_nominal_theta_lambda:
        plant_nom = collect_plant_theta_lambda(cfg, gamma=cfg.gamma, g_phi=cfg.g_phi, dataset_name="plant_nominal")
        informative = extract_informative_thetas(plant_nom)
        print("Informative thetas:", informative, flush=True)

    if cfg.do_qco_theta_lambda:
        if plant_nom is None:
            plant_nom = collect_plant_theta_lambda(cfg, gamma=cfg.gamma, g_phi=cfg.g_phi, dataset_name="plant_nominal")
            informative = extract_informative_thetas(plant_nom)
        qco_nom = collect_qco_vs_cached_plant(cfg, plant_nom, gamma_s=cfg.gamma_s, g_phi=cfg.g_phi, dataset_name="qco_nominal")
        plot_nominal_maps(plant_nom, qco_nom, cfg, prefix="nominal")

    if cfg.do_fit_uncertainty:
        if plant_nom is None:
            plant_nom = collect_plant_theta_lambda(cfg, gamma=cfg.gamma, g_phi=cfg.g_phi, dataset_name="plant_nominal")
            informative = extract_informative_thetas(plant_nom)
        fit_nom = analyze_fit_global_and_families(plant_nom, cfg, prefix="plant_nominal")
        plot_fit_invrootJ(plant_nom, fit_nom, cfg, prefix="nominal")
        print("[fit global]", fit_nom["global_fit"], flush=True)
        print("[fit global CI]", fit_nom["global_ci"], flush=True)

    if informative is None:
        informative = extract_informative_thetas(plant_nom)
    theta_list = informative["theta_list"]

    if cfg.do_gamma_strategic:
        gamma_summary = strategic_parameter_sweep(
            cfg, param_name="gamma", param_vals=np.asarray(cfg.gamma_vals, dtype=float),
            theta_list=theta_list, lambda_vals=np.asarray(cfg.lambda_vals, dtype=float),
            prefix="gamma"
        )
        plot_strategic_sweep(gamma_summary, cfg, param_symbol=r"$\gamma$", prefix="gamma")

    if cfg.do_gphi_strategic:
        gphi_summary = strategic_parameter_sweep(
            cfg, param_name="g_phi", param_vals=np.asarray(cfg.gphi_vals, dtype=float),
            theta_list=theta_list, lambda_vals=np.asarray(cfg.lambda_vals, dtype=float),
            prefix="gphi"
        )
        plot_strategic_sweep(gphi_summary, cfg, param_symbol=r"$g_\phi$", prefix="gphi")

    print(f"Done. Outputs saved in {cfg.save_dir}", flush=True)


if __name__ == "__main__":
    main()
