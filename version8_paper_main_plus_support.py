
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import hashlib
from dataclasses import dataclass, field
import os

import matplotlib
matplotlib.use("Agg")

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
_DEFAULT_C_PS = 1.0
_DEFAULT_C_PO = 0.4

@dataclass
class RunConfig:
    save_dir: str = "plots_paper_plus_support"
    mode: str = "DEBUG"
    n_jobs: int = max(1, (os.cpu_count() or 2) - 1)
    dt: float = 0.01
    T: float = 20.0
    n_mc: int = 6
    burn_frac: float = 0.2
    gamma: float = 1.0
    omega: float = 0.0
    g_phi: float = 2.0
    kappa: float = 1.0
    q_phi: float = 0.01
    meas_std: float = 0.03
    sigma_q: float = 0.02
    sigma_p: float = 0.02
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
    c_ps: float = _DEFAULT_C_PS
    c_qo: float = 0.0
    c_po: float = _DEFAULT_C_PO
    theta_vals: np.ndarray = field(default_factory=lambda: np.linspace(0.05, np.pi - 0.05, 15))
    lambda_vals: np.ndarray = field(default_factory=lambda: np.linspace(0.02, 1.5, 12))
    gamma_vals: np.ndarray = field(default_factory=lambda: np.array([0.5, 1.0, 2.0]))
    gphi_vals: np.ndarray = field(default_factory=lambda: np.array([0.5, 1.0, 2.0]))
    make_main_figures: bool = True
    make_support_fit_figures: bool = True
    make_support_param_sweeps: bool = True
    make_qco_maps: bool = True
    make_qco_scatter: bool = True
    show_family_fit: bool = True
    show_family_uncertainty: bool = True
    n_boot: int = 200

    def __post_init__(self):
        mode = self.mode.upper()
        if mode == "INTERMEDIATE":
            self.T = 35.0
            self.n_mc = 8
            self.theta_vals = np.linspace(0.05, np.pi - 0.05, 25)
            self.lambda_vals = np.linspace(0.02, 1.5, 20)
            self.gamma_vals = np.array([0.5, 1.0, 1.5, 2.0])
            self.gphi_vals = np.array([0.5, 1.0, 2.0, 3.0])
            self.n_boot = 300
        elif mode == "FINAL":
            self.T = 50.0
            self.n_mc = 12
            self.theta_vals = np.linspace(0.05, np.pi - 0.05, 35)
            self.lambda_vals = np.linspace(0.02, 1.5, 30)
            self.gamma_vals = np.array([0.3, 0.5, 1.0, 1.5, 2.0, 3.0])
            self.gphi_vals = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
            self.n_boot = 500

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def checkpoint(msg: str):
    print(f"\n=== {msg} ===", flush=True)

def save_figure(fig, save_dir: str, fname: str, dpi: int = 320):
    ensure_dir(save_dir)
    fig.savefig(Path(save_dir) / f"{fname}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(Path(save_dir) / f"{fname}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

def save_json(save_dir: str, fname: str, payload: dict):
    ensure_dir(save_dir)
    with open(Path(save_dir) / f"{fname}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def save_npz(save_dir: str, fname: str, **arrays):
    ensure_dir(save_dir)
    np.savez(Path(save_dir) / f"{fname}.npz", **arrays)

def setup_axes(ax, xlabel=None, ylabel=None, title=None, grid=True):
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=13, pad=10)
    if grid:
        ax.grid(True, alpha=0.25, linewidth=0.6)

def stable_seed(*vals) -> int:
    key = "_".join([f"{float(v):.12e}" for v in vals])
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)

def safe_inv_sqrt(x, eps=EPS):
    x = np.asarray(x, dtype=float)
    return 1.0 / np.sqrt(np.maximum(x, eps))

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

def gramian_horizon(gamma: float, dt: float, n_decay: float = 5.0, floor: int = 20, cap: int = 200) -> int:
    steps = int(np.ceil(n_decay / max(gamma * dt, EPS)))
    return max(floor, min(steps, cap))

def run_parallel(tasks, worker_fn, n_jobs=1, desc=""):
    if desc:
        print(f"Running: {desc} | #tasks={len(tasks)} | n_jobs={n_jobs}", flush=True)
    if n_jobs == 1:
        return [worker_fn(*t) for t in tasks]
    return Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(delayed(worker_fn)(*t) for t in tasks)

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
    n = F.shape[0]
    e_phi = np.zeros(n, dtype=float)
    e_phi[phase_index] = 1.0
    Fk = np.eye(n)
    J_phi = 0.0
    for _ in range(horizon):
        out = H @ (Fk @ e_phi)
        J_phi += float(np.dot(out.ravel(), out.ravel()))
        Fk = F @ Fk
    return float(J_phi)

def gramian_metrics(F: np.ndarray, H: np.ndarray, horizon: int, phase_index: int) -> dict:
    Wo = discrete_observability_gramian(F, H, horizon)
    eigvals = np.maximum(np.linalg.eigvalsh(Wo), 0.0)
    lam_min = float(np.min(eigvals))
    lam_max = float(np.max(eigvals))
    cond = np.inf if lam_min < 1e-14 else float(lam_max / lam_min)
    J_phi = phase_observability_metric(F, H, phase_index=phase_index, horizon=horizon)
    return {"lambda_min": lam_min, "lambda_max": lam_max, "cond": cond, "trace": float(np.trace(Wo)), "J_phi": J_phi}

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
    return Ks

def batched_mc_phase_stats(F, H, Qd, R, x_true0, x_hat0, P0, T, dt, phase_index, n_mc=8, burn_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    n_steps = int(np.round(T / dt))
    burn = int(burn_frac * n_steps)
    n = F.shape[0]
    Hvec = H.reshape(-1)
    Ks = precompute_kalman_gains(F, H, Qd, R, P0, n_steps)
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
    rmse_mean, rmse_std, rmse_sem = mean_std_sem(rmse_per_run)
    q16, q84 = quantile_interval(rmse_per_run)
    return {"rmse_mean": rmse_mean, "rmse_std": rmse_std, "rmse_sem": rmse_sem, "rmse_q16": q16, "rmse_q84": q84}

def build_plant_system(theta, lam, cfg: RunConfig, gamma=None, g_phi=None):
    gamma = cfg.gamma if gamma is None else gamma
    g_phi = cfg.g_phi if g_phi is None else g_phi
    A = build_Aa(gamma=gamma, omega=cfg.omega, g_phi=g_phi, lam=lam)
    H = build_Ca(kappa=cfg.kappa, theta=theta)
    Qc = build_continuous_process_covariance(sigma_q=cfg.sigma_q, sigma_p=cfg.sigma_p, q_phi=cfg.q_phi)
    F, Qd = discretize_plant(A, Qc, cfg.dt)
    R = np.array([[cfg.meas_std ** 2]], dtype=float)
    return F, H, Qd, R

def build_qco_system(theta, lam, cfg: RunConfig, gamma_s=None, g_phi=None, k_so=None, k_os=None):
    gamma_s = cfg.gamma_s if gamma_s is None else gamma_s
    g_phi = cfg.g_phi if g_phi is None else g_phi
    k_so = cfg.k_so if k_so is None else k_so
    k_os = cfg.k_os if k_os is None else k_os
    A = build_qco_augmented_A(gamma_s=gamma_s, omega_s=cfg.omega_s, gamma_o=cfg.gamma_o, omega_o=cfg.omega_o, g_phi=g_phi, lam=lam, k_so=k_so, k_os=k_os)
    H = build_qco_measurement(c_qs=cfg.c_qs, c_ps=cfg.c_ps, c_qo=cfg.c_qo, c_po=cfg.c_po)
    Qc = build_qco_process_covariance(sigma_qs=cfg.sigma_qs, sigma_ps=cfg.sigma_ps, sigma_qo=cfg.sigma_qo, sigma_po=cfg.sigma_po, q_phi=cfg.q_phi)
    F, Qd = discretize_qco(A, Qc, cfg.dt)
    R = np.array([[cfg.meas_std ** 2]], dtype=float)
    return F, H, Qd, R

def plant_metrics(theta, lam, cfg: RunConfig, gamma=None, g_phi=None):
    gamma = cfg.gamma if gamma is None else gamma
    g_phi = cfg.g_phi if g_phi is None else g_phi
    F, H, Qd, R = build_plant_system(theta, lam, cfg, gamma=gamma, g_phi=g_phi)
    x_true0 = np.array([0.0, 0.0, 0.5], dtype=float)
    x_hat0 = np.zeros(3, dtype=float)
    P0 = 10.0 * np.eye(3)
    stats = batched_mc_phase_stats(F, H, Qd, R, x_true0, x_hat0, P0, T=cfg.T, dt=cfg.dt, phase_index=2, n_mc=cfg.n_mc, burn_frac=cfg.burn_frac, seed=stable_seed(theta, lam, gamma, g_phi, cfg.q_phi, cfg.meas_std))
    gram = gramian_metrics(F, H, gramian_horizon(gamma, cfg.dt), phase_index=2)
    return {"F": F, "H": H, "Qd": Qd, "R": R, "rmse_phi": stats["rmse_mean"], "rmse_std": stats["rmse_std"], "rmse_sem": stats["rmse_sem"], "rmse_q16": stats["rmse_q16"], "rmse_q84": stats["rmse_q84"], "noise_proxy": float(np.trace(Qd)), **gram}

def qco_metrics(theta, lam, cfg: RunConfig, gamma_s=None, g_phi=None, k_so=None, k_os=None):
    gamma_s = cfg.gamma_s if gamma_s is None else gamma_s
    g_phi = cfg.g_phi if g_phi is None else g_phi
    k_so = cfg.k_so if k_so is None else k_so
    k_os = cfg.k_os if k_os is None else k_os
    F, H, Qd, R = build_qco_system(theta, lam, cfg, gamma_s=gamma_s, g_phi=g_phi, k_so=k_so, k_os=k_os)
    x_true0 = np.array([0.0, 0.0, 0.0, 0.0, 0.5], dtype=float)
    x_hat0 = np.zeros(5, dtype=float)
    P0 = 10.0 * np.eye(5)
    stats = batched_mc_phase_stats(F, H, Qd, R, x_true0, x_hat0, P0, T=cfg.T, dt=cfg.dt, phase_index=4, n_mc=cfg.n_mc, burn_frac=cfg.burn_frac, seed=stable_seed(theta, lam, gamma_s, g_phi, cfg.q_phi, cfg.meas_std, k_so, k_os))
    gram = gramian_metrics(F, H, gramian_horizon(gamma_s, cfg.dt), phase_index=4)
    return {"F": F, "H": H, "Qd": Qd, "R": R, "rmse_phi": stats["rmse_mean"], "rmse_std": stats["rmse_std"], "rmse_sem": stats["rmse_sem"], "rmse_q16": stats["rmse_q16"], "rmse_q84": stats["rmse_q84"], "noise_proxy": float(np.trace(Qd)), **gram}

def collect_plant_theta_lambda_data(cfg: RunConfig, gamma=None, g_phi=None, name="plant_nominal"):
    theta_vals = np.asarray(cfg.theta_vals, dtype=float)
    lambda_vals = np.asarray(cfg.lambda_vals, dtype=float)
    tasks = [(i, j, lam, theta) for i, lam in enumerate(lambda_vals) for j, theta in enumerate(theta_vals)]
    def _run(i, j, lam, theta):
        met = plant_metrics(theta=theta, lam=lam, cfg=cfg, gamma=gamma, g_phi=g_phi)
        return i, j, met
    results = run_parallel(tasks, _run, n_jobs=cfg.n_jobs, desc=f"collect_plant_theta_lambda[{name}]")
    shape = (len(lambda_vals), len(theta_vals))
    rmse_map = np.zeros(shape); rmse_sem_map = np.zeros(shape); rmse_q16_map = np.zeros(shape); rmse_q84_map = np.zeros(shape)
    jphi_map = np.zeros(shape); cond_map = np.zeros(shape); lambda_min_map = np.zeros(shape)
    rows = []; cache = {}
    for i, j, met in results:
        lam = lambda_vals[i]; theta = theta_vals[j]
        rmse_map[i, j] = met["rmse_phi"]; rmse_sem_map[i, j] = met["rmse_sem"]; rmse_q16_map[i, j] = met["rmse_q16"]; rmse_q84_map[i, j] = met["rmse_q84"]
        jphi_map[i, j] = met["J_phi"]; cond_map[i, j] = met["cond"]; lambda_min_map[i, j] = met["lambda_min"]
        rows.append({"theta": theta, "lambda": lam, "rmse_phi": met["rmse_phi"], "rmse_sem": met["rmse_sem"], "rmse_q16": met["rmse_q16"], "rmse_q84": met["rmse_q84"], "J_phi": met["J_phi"], "inv_sqrt_Jphi": 1.0 / np.sqrt(max(met["J_phi"], EPS)), "cond": met["cond"], "lambda_min": met["lambda_min"]})
        cache[(i, j)] = met
    save_npz(cfg.save_dir, name, theta_vals=theta_vals, lambda_vals=lambda_vals, rmse_map=rmse_map, rmse_sem_map=rmse_sem_map, rmse_q16_map=rmse_q16_map, rmse_q84_map=rmse_q84_map, jphi_map=jphi_map, cond_map=cond_map, lambda_min_map=lambda_min_map)
    return {"theta_vals": theta_vals, "lambda_vals": lambda_vals, "rmse_map": rmse_map, "rmse_sem_map": rmse_sem_map, "rmse_q16_map": rmse_q16_map, "rmse_q84_map": rmse_q84_map, "jphi_map": jphi_map, "cond_map": cond_map, "lambda_min_map": lambda_min_map, "rows": rows, "cache": cache}

def compare_plant_vs_qco(theta, lam, cfg: RunConfig, plant_cached=None, gamma_s=None, g_phi=None, k_so=None, k_os=None):
    if plant_cached is None:
        plant = plant_metrics(theta, lam, cfg, gamma=gamma_s if gamma_s is not None else cfg.gamma, g_phi=g_phi)
    else:
        plant = plant_cached
    qco = qco_metrics(theta, lam, cfg, gamma_s=gamma_s, g_phi=g_phi, k_so=k_so, k_os=k_os)
    delta_rmse = (plant["rmse_phi"] - qco["rmse_phi"]) / max(abs(plant["rmse_phi"]), EPS)
    delta_J_rel = (qco["J_phi"] - plant["J_phi"]) / max(abs(plant["J_phi"]), EPS)
    delta_J_log = float(np.log(qco["J_phi"] + EPS) - np.log(plant["J_phi"] + EPS))
    delta_Q = (qco["noise_proxy"] - plant["noise_proxy"]) / max(abs(plant["noise_proxy"]), EPS)
    return {"plant": plant, "qco": qco, "delta_rmse": delta_rmse, "delta_Jphi": delta_J_rel, "delta_Jphi_log": delta_J_log, "delta_Q": delta_Q}

def regime_map_theta_lambda(cfg: RunConfig, plant_data: dict, name="theta_lambda_nominal"):
    theta_vals = np.asarray(plant_data["theta_vals"], dtype=float)
    lambda_vals = np.asarray(plant_data["lambda_vals"], dtype=float)
    tasks = [(i, j, lam, theta) for i, lam in enumerate(lambda_vals) for j, theta in enumerate(theta_vals)]
    def _run(i, j, lam, theta):
        comp = compare_plant_vs_qco(theta=theta, lam=lam, cfg=cfg, plant_cached=plant_data["cache"][(i, j)], gamma_s=cfg.gamma_s, g_phi=cfg.g_phi, k_so=cfg.k_so, k_os=cfg.k_os)
        return i, j, comp
    results = run_parallel(tasks, _run, n_jobs=cfg.n_jobs, desc=f"regime_map_theta_lambda[{name}]")
    shape = (len(lambda_vals), len(theta_vals))
    delta_rmse = np.zeros(shape); delta_Jphi = np.zeros(shape); delta_Jphi_log = np.zeros(shape); delta_Q = np.zeros(shape)
    for i, j, comp in results:
        delta_rmse[i, j] = comp["delta_rmse"]; delta_Jphi[i, j] = comp["delta_Jphi"]; delta_Jphi_log[i, j] = comp["delta_Jphi_log"]; delta_Q[i, j] = comp["delta_Q"]
    save_npz(cfg.save_dir, name, theta_vals=theta_vals, lambda_vals=lambda_vals, delta_rmse=delta_rmse, delta_Jphi=delta_Jphi, delta_Jphi_log=delta_Jphi_log, delta_Q=delta_Q)
    return {"theta_vals": theta_vals, "lambda_vals": lambda_vals, "delta_rmse": delta_rmse, "delta_Jphi": delta_Jphi, "delta_Jphi_log": delta_Jphi_log, "delta_Q": delta_Q}

def fit_affine_invrootJ(Jphi, rmse):
    x = safe_inv_sqrt(Jphi).ravel(); y = np.asarray(rmse, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y); x = x[m]; y = y[m]
    if len(x) < 3 or np.std(x) < EPS:
        return {"beta": None, "r2": np.nan, "fit_rmse": np.nan, "n": len(x)}
    X = np.column_stack([x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2)); ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot < EPS else 1.0 - ss_res / ss_tot
    fit_rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return {"beta": beta, "r2": r2, "fit_rmse": fit_rmse, "n": len(x)}

def fit_affine_invrootJ_plus_lambda(Jphi, lam, rmse):
    x1 = safe_inv_sqrt(Jphi).ravel(); x2 = np.asarray(lam, dtype=float).ravel(); y = np.asarray(rmse, dtype=float).ravel()
    m = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y); x1 = x1[m]; x2 = x2[m]; y = y[m]
    if len(x1) < 4:
        return {"beta": None, "r2": np.nan, "fit_rmse": np.nan, "n": len(x1)}
    X = np.column_stack([x1, x2, np.ones_like(x1)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2)); ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot < EPS else 1.0 - ss_res / ss_tot
    fit_rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return {"beta": beta, "r2": r2, "fit_rmse": fit_rmse, "n": len(x1)}

def bootstrap_fit_ci(Jphi, rmse, lam=None, n_boot=200, seed=0):
    x = safe_inv_sqrt(Jphi).ravel(); y = np.asarray(rmse, dtype=float).ravel()
    z = None if lam is None else np.asarray(lam, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y) if z is None else np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[mask]; y = y[mask]; z = None if z is None else z[mask]
    n = len(x)
    if n < 5:
        return {}
    rng = np.random.default_rng(seed)
    betas = []; r2s = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        X = np.column_stack([x[idx], np.ones(n)]) if z is None else np.column_stack([x[idx], z[idx], np.ones(n)])
        beta, *_ = np.linalg.lstsq(X, y[idx], rcond=None)
        yhat = X @ beta
        ss_res = float(np.sum((y[idx] - yhat) ** 2)); ss_tot = float(np.sum((y[idx] - np.mean(y[idx])) ** 2))
        r2 = np.nan if ss_tot < EPS else 1.0 - ss_res / ss_tot
        betas.append(beta); r2s.append(r2)
    betas = np.asarray(betas, dtype=float)
    return {"beta_q16": np.quantile(betas, 0.16, axis=0).tolist(), "beta_q84": np.quantile(betas, 0.84, axis=0).tolist(), "r2_q16": float(np.quantile(r2s, 0.16)), "r2_q84": float(np.quantile(r2s, 0.84))}

def analyze_noncollapse_quantitatively(data_dict, cfg: RunConfig, prefix="nominal"):
    theta = np.array([r["theta"] for r in data_dict["rows"]], dtype=float)
    lam = np.array([r["lambda"] for r in data_dict["rows"]], dtype=float)
    rmse = np.array([r["rmse_phi"] for r in data_dict["rows"]], dtype=float)
    Jphi = np.array([r["J_phi"] for r in data_dict["rows"]], dtype=float)
    fit1 = fit_affine_invrootJ(Jphi, rmse); fit2 = fit_affine_invrootJ_plus_lambda(Jphi, lam, rmse)
    ci1 = bootstrap_fit_ci(Jphi, rmse, lam=None, n_boot=cfg.n_boot, seed=stable_seed(0.1, 0.2))
    ci2 = bootstrap_fit_ci(Jphi, rmse, lam=lam, n_boot=cfg.n_boot, seed=stable_seed(0.3, 0.4))
    x = safe_inv_sqrt(Jphi)
    beta1 = fit1["beta"]; beta2 = fit2["beta"]
    resid1 = rmse - (beta1[0] * x + beta1[1]) if beta1 is not None else np.full_like(rmse, np.nan)
    resid2 = rmse - (beta2[0] * x + beta2[1] * lam + beta2[2]) if beta2 is not None else np.full_like(rmse, np.nan)
    summary = {"fit_invrootJ_plus_b": {"beta": None if fit1["beta"] is None else fit1["beta"].tolist(), "r2": fit1["r2"], "fit_rmse": fit1["fit_rmse"], "n": fit1["n"], "ci": ci1}, "fit_invrootJ_plus_c_lambda_plus_b": {"beta": None if fit2["beta"] is None else fit2["beta"].tolist(), "r2": fit2["r2"], "fit_rmse": fit2["fit_rmse"], "n": fit2["n"], "ci": ci2}, "residual_std_model1": float(np.nanstd(resid1)), "residual_std_model2": float(np.nanstd(resid2))}
    save_json(cfg.save_dir, f"{prefix}_noncollapse_quant", summary)
    return {"summary": summary, "resid1": resid1, "resid2": resid2, "theta": theta, "lambda": lam, "rmse": rmse, "Jphi": Jphi}

def plot_jphi_metric_map(data_dict, cfg: RunConfig, prefix="nominal"):
    TH, LA = np.meshgrid(data_dict["theta_vals"], data_dict["lambda_vals"]); Z = data_dict["jphi_map"]
    fig, ax = plt.subplots(figsize=(8.4, 5.5), constrained_layout=True)
    vmin, vmax = robust_positive_limits(Z)
    im = ax.pcolormesh(TH, LA, Z, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, pad=0.02); cbar.set_label(r"$J_\phi(N)$", fontsize=11)
    setup_axes(ax, xlabel=r"Homodyne angle $\theta$", ylabel=r"OU rate $\lambda$", title="Directional phase metric map", grid=False)
    save_figure(fig, cfg.save_dir, f"{prefix}_jphi_metric_map")

def plot_family_fixed_theta_with_uncertainty(data_dict, cfg: RunConfig, prefix="nominal", theta_targets=None, x_clip=0.75):
    theta_vals = data_dict["theta_vals"]; rmse_map = data_dict["rmse_map"]; rmse_q16_map = data_dict["rmse_q16_map"]; rmse_q84_map = data_dict["rmse_q84_map"]; jphi_map = data_dict["jphi_map"]
    if theta_targets is None:
        theta_targets = [0.2, 0.6, np.pi / 2, 2.2, 2.9]
    theta_indices = [int(np.argmin(np.abs(theta_vals - t))) for t in theta_targets]
    seen = set(); theta_indices = [idx for idx in theta_indices if not (idx in seen or seen.add(idx))]
    fig, ax = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    fit_rows = []
    for idx in theta_indices:
        th = theta_vals[idx]
        x = safe_inv_sqrt(jphi_map[:, idx]); y = rmse_map[:, idx]; ylo = rmse_q16_map[:, idx]; yhi = rmse_q84_map[:, idx]
        mask = np.isfinite(x) & np.isfinite(y)
        if x_clip is not None:
            mask &= (x <= x_clip)
        x = x[mask]; y = y[mask]; ylo = ylo[mask]; yhi = yhi[mask]
        if len(x) < 2:
            continue
        order = np.argsort(x); x = x[order]; y = y[order]; ylo = ylo[order]; yhi = yhi[order]
        ax.plot(x, y, marker="o", markersize=4, linewidth=2.0, label=fr"$\theta={th:.2f}$")
        if cfg.show_family_uncertainty:
            ax.fill_between(x, ylo, yhi, alpha=0.18)
        if cfg.show_family_fit and len(x) >= 4:
            X = np.column_stack([x, np.ones_like(x)]); beta, *_ = np.linalg.lstsq(X, y, rcond=None); yhat = X @ beta
            ax.plot(x, yhat, linewidth=1.2, linestyle="--", alpha=0.9)
            fit_rows.append({"theta": float(th), "a": float(beta[0]), "b": float(beta[1]), "n": int(len(x))})
    setup_axes(ax, xlabel=r"$1/\sqrt{J_\phi(N)}$", ylabel=r"$\mathrm{RMSE}(\phi)$", title=r"Local families at fixed $\theta$")
    if x_clip is not None:
        ax.set_xlim(left=0.0, right=x_clip * 1.05)
    ax.legend(frameon=True, fontsize=9)
    save_figure(fig, cfg.save_dir, f"{prefix}_families_fixed_theta_uncertainty")
    save_json(cfg.save_dir, f"{prefix}_families_fixed_theta_fits", {"rows": fit_rows})

def plot_global_noncollapse_colored_by_lambda(data_dict, cfg: RunConfig, prefix="nominal"):
    rows = data_dict["rows"]
    lam = np.array([r["lambda"] for r in rows], dtype=float); rmse = np.array([r["rmse_phi"] for r in rows], dtype=float); Jphi = np.array([r["J_phi"] for r in rows], dtype=float); invsqrt = safe_inv_sqrt(Jphi)
    fig, ax = plt.subplots(figsize=(6.8, 5.2), constrained_layout=True)
    sc = ax.scatter(invsqrt, rmse, c=lam, s=16, alpha=0.6)
    fig.colorbar(sc, ax=ax, label=r"$\lambda$")
    setup_axes(ax, xlabel=r"$1/\sqrt{J_\phi(N)}$", ylabel=r"$\mathrm{RMSE}(\phi)$", title="Global non-collapse colored by $\lambda$")
    save_figure(fig, cfg.save_dir, f"{prefix}_noncollapse_global_lambda")

def plot_qco_advantage_map(qco_dict, cfg: RunConfig, prefix="nominal"):
    TH, LA = np.meshgrid(qco_dict["theta_vals"], qco_dict["lambda_vals"]); Z = qco_dict["delta_rmse"]
    fig, ax = plt.subplots(figsize=(8.4, 5.5), constrained_layout=True)
    vmin, vmax = robust_signed_limits(Z)
    im = ax.pcolormesh(TH, LA, Z, shading="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax))
    cbar = fig.colorbar(im, ax=ax, pad=0.02); cbar.set_label(r"$\Delta_{\rm RMSE}$", fontsize=11)
    try:
        cs = ax.contour(TH, LA, Z, levels=[0.0], colors="k", linewidths=1.2); ax.clabel(cs, fmt={0.0: "0"}, fontsize=9)
    except Exception:
        pass
    setup_axes(ax, xlabel=r"Homodyne angle $\theta$", ylabel=r"OU rate $\lambda$", title="QCO advantage map", grid=False)
    save_figure(fig, cfg.save_dir, f"{prefix}_qco_advantage_map")

def plot_qco_delta_rmse_vs_delta_jphi(qco_dict, cfg: RunConfig, prefix="nominal", x_abs_clip=10.0, n_bins=25, min_count_per_bin=8):
    x = qco_dict["delta_Jphi"].ravel().astype(float); y = qco_dict["delta_rmse"].ravel().astype(float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (x <= x_abs_clip); x = x[mask]; y = y[mask]
    if len(x) < 10:
        return
    bins = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), n_bins + 1); centers = np.sqrt(bins[:-1] * bins[1:])
    y_mean = np.full(n_bins, np.nan); y_std = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = (x >= bins[i]) & (x < bins[i + 1]) if i < n_bins - 1 else (x >= bins[i]) & (x <= bins[i + 1])
        if np.sum(m) >= min_count_per_bin:
            y_mean[i] = np.mean(y[m]); y_std[i] = np.std(y[m])
    valid = np.isfinite(y_mean)
    fig, ax = plt.subplots(figsize=(6.8, 5.2), constrained_layout=True)
    ax.scatter(x, y, s=10, alpha=0.15, label="simulation points")
    if np.any(valid):
        ax.plot(centers[valid], y_mean[valid], linewidth=2.5, label="binned mean")
        ax.fill_between(centers[valid], y_mean[valid] - y_std[valid], y_mean[valid] + y_std[valid], alpha=0.2)
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="k", alpha=0.7); ax.axvline(0.0, linestyle="--", linewidth=1.0, color="k", alpha=0.7)
    ax.set_xscale("log")
    setup_axes(ax, xlabel=r"$\Delta_{J_\phi}^{\mathrm{rel}}$", ylabel=r"$\Delta_{\mathrm{RMSE}}$", title="QCO performance vs phase-information gain", grid=True)
    ax.legend()
    save_figure(fig, cfg.save_dir, f"{prefix}_delta_rmse_vs_delta_jphi")

def plot_support_global_fit(noncollapse_dict, cfg: RunConfig, prefix="support"):
    J = noncollapse_dict["Jphi"]; rmse = noncollapse_dict["rmse"]; lam = noncollapse_dict["lambda"]; x = safe_inv_sqrt(J)
    fit1 = noncollapse_dict["summary"]["fit_invrootJ_plus_b"]["beta"]
    fig, ax = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)
    sc = ax.scatter(x, rmse, c=lam, s=16, alpha=0.75); fig.colorbar(sc, ax=ax, label=r"$\lambda$")
    if fit1 is not None:
        xx = np.linspace(np.min(x), np.max(x), 200); yy = fit1[0] * xx + fit1[1]
        ax.plot(xx, yy, linewidth=2.2, label=r"$a/\sqrt{J_\phi}+b$")
    setup_axes(ax, xlabel=r"$1/\sqrt{J_\phi(N)}$", ylabel=r"$\mathrm{RMSE}(\phi)$", title="Support: global affine inverse-root fit"); ax.legend()
    save_figure(fig, cfg.save_dir, f"{prefix}_fit_invrootJ_plus_b")
    resid1 = noncollapse_dict["resid1"]
    fig, ax = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)
    sc = ax.scatter(x, resid1, c=lam, s=16, alpha=0.75); fig.colorbar(sc, ax=ax, label=r"$\lambda$")
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="k", alpha=0.7)
    setup_axes(ax, xlabel=r"$1/\sqrt{J_\phi(N)}$", ylabel="Residual", title=r"Support: residuals of $a/\sqrt{J_\phi}+b$")
    save_figure(fig, cfg.save_dir, f"{prefix}_residuals_invrootJ_plus_b")
    resid2 = noncollapse_dict["resid2"]
    fig, ax = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)
    sc = ax.scatter(x, resid2, c=lam, s=16, alpha=0.75); fig.colorbar(sc, ax=ax, label=r"$\lambda$")
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="k", alpha=0.7)
    setup_axes(ax, xlabel=r"$1/\sqrt{J_\phi(N)}$", ylabel="Residual", title=r"Support: residuals of $a/\sqrt{J_\phi}+c\lambda+b$")
    save_figure(fig, cfg.save_dir, f"{prefix}_residuals_invrootJ_plus_c_lambda_plus_b")

def strategic_param_sweep(cfg: RunConfig, param_name: str, param_vals: np.ndarray, theta_list, lambda_vals):
    tasks = [(float(p), float(theta), float(lam)) for p in param_vals for theta in theta_list for lam in lambda_vals]
    def _one(p, theta, lam):
        if param_name == "gamma":
            plant = plant_metrics(theta, lam, cfg, gamma=p, g_phi=cfg.g_phi)
            qco = qco_metrics(theta, lam, cfg, gamma_s=p, g_phi=cfg.g_phi)
        elif param_name == "g_phi":
            plant = plant_metrics(theta, lam, cfg, gamma=cfg.gamma, g_phi=p)
            qco = qco_metrics(theta, lam, cfg, gamma_s=cfg.gamma_s, g_phi=p)
        else:
            raise ValueError("Unknown param_name")
        delta_rmse = (plant["rmse_phi"] - qco["rmse_phi"]) / max(abs(plant["rmse_phi"]), EPS)
        delta_Jphi = (qco["J_phi"] - plant["J_phi"]) / max(abs(plant["J_phi"]), EPS)
        return {"param": p, "theta": theta, "lambda": lam, "delta_rmse": delta_rmse, "delta_Jphi": delta_Jphi}
    raw = run_parallel(tasks, _one, n_jobs=cfg.n_jobs, desc=f"strategic_{param_name}")
    rows = []
    for p in param_vals:
        for theta in theta_list:
            sub = [r for r in raw if abs(r["param"] - p) < 1e-12 and abs(r["theta"] - theta) < 1e-12]
            d_rmse = np.array([r["delta_rmse"] for r in sub], dtype=float); d_j = np.array([r["delta_Jphi"] for r in sub], dtype=float)
            m_rmse, s_rmse, sem_rmse = mean_std_sem(d_rmse); q16_rmse, q84_rmse = quantile_interval(d_rmse); m_j, s_j, sem_j = mean_std_sem(d_j)
            rows.append({"param": float(p), "theta": float(theta), "delta_rmse_mean": m_rmse, "delta_rmse_sem": sem_rmse, "delta_rmse_q16": q16_rmse, "delta_rmse_q84": q84_rmse, "delta_Jphi_mean": m_j, "delta_Jphi_sem": sem_j})
    return rows

def plot_support_param_sweep(rows, cfg: RunConfig, param_symbol=r"$\gamma$", prefix="gamma"):
    thetas = sorted({round(r["theta"], 12) for r in rows})
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for th in thetas:
        sub = [r for r in rows if abs(r["theta"] - th) < 1e-12]
        p = np.array([r["param"] for r in sub], dtype=float); m = np.array([r["delta_rmse_mean"] for r in sub], dtype=float); lo = np.array([r["delta_rmse_q16"] for r in sub], dtype=float); hi = np.array([r["delta_rmse_q84"] for r in sub], dtype=float)
        ax.plot(p, m, marker="o", linewidth=2.0, label=rf"$\theta={th:.2f}$"); ax.fill_between(p, lo, hi, alpha=0.16)
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="k", alpha=0.7)
    setup_axes(ax, xlabel=param_symbol, ylabel=r"mean $\Delta_{\rm RMSE}$ over $\lambda$", title=rf"Support: QCO gain vs {param_symbol}"); ax.legend(fontsize=8)
    save_figure(fig, cfg.save_dir, f"support_{prefix}_delta_rmse")
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for th in thetas:
        sub = [r for r in rows if abs(r["theta"] - th) < 1e-12]
        p = np.array([r["param"] for r in sub], dtype=float); m = np.array([r["delta_Jphi_mean"] for r in sub], dtype=float); sem = np.array([r["delta_Jphi_sem"] for r in sub], dtype=float)
        ax.plot(p, m, marker="s", linewidth=2.0, label=rf"$\theta={th:.2f}$"); ax.fill_between(p, m - sem, m + sem, alpha=0.16)
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="k", alpha=0.7)
    setup_axes(ax, xlabel=param_symbol, ylabel=r"mean $\Delta J_\phi$ over $\lambda$", title=rf"Support: phase-information gain vs {param_symbol}"); ax.legend(fontsize=8)
    save_figure(fig, cfg.save_dir, f"support_{prefix}_delta_Jphi")

def main():
    cfg = RunConfig()
    ensure_dir(cfg.save_dir)
    checkpoint(f"Mode = {cfg.mode}")
    checkpoint("Collecting nominal plant data")
    plant_data = collect_plant_theta_lambda_data(cfg, gamma=cfg.gamma, g_phi=cfg.g_phi, name="plant_nominal")
    checkpoint("Collecting nominal QCO theta-lambda comparison")
    qco_data = regime_map_theta_lambda(cfg, plant_data, name="qco_nominal")
    checkpoint("Quantifying non-collapse")
    noncollapse = analyze_noncollapse_quantitatively(plant_data, cfg, prefix="nominal")
    checkpoint("Making main paper figures")
    plot_jphi_metric_map(plant_data, cfg, prefix="main")
    plot_family_fixed_theta_with_uncertainty(plant_data, cfg, prefix="main", x_clip=0.75)
    plot_global_noncollapse_colored_by_lambda(plant_data, cfg, prefix="main")
    plot_qco_advantage_map(qco_data, cfg, prefix="main")
    plot_qco_delta_rmse_vs_delta_jphi(qco_data, cfg, prefix="main")
    checkpoint("Making support fit/residual figures")
    plot_support_global_fit(noncollapse, cfg, prefix="support")
    checkpoint("Making support strategic parameter sweeps")
    theta_list = [0.6, float(np.pi / 2), 2.2]
    rows_gamma = strategic_param_sweep(cfg, param_name="gamma", param_vals=np.asarray(cfg.gamma_vals, dtype=float), theta_list=theta_list, lambda_vals=np.asarray(cfg.lambda_vals, dtype=float))
    plot_support_param_sweep(rows_gamma, cfg, param_symbol=r"$\gamma$", prefix="gamma")
    rows_gphi = strategic_param_sweep(cfg, param_name="g_phi", param_vals=np.asarray(cfg.gphi_vals, dtype=float), theta_list=theta_list, lambda_vals=np.asarray(cfg.lambda_vals, dtype=float))
    plot_support_param_sweep(rows_gphi, cfg, param_symbol=r"$g_\phi$", prefix="gphi")
    checkpoint("Done")
    print("All figures and summaries saved to:", cfg.save_dir, flush=True)

if __name__ == "__main__":
    main()
