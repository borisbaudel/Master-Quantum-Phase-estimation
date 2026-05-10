"""
Quantum Coherent Luenberger Observer
Version simple avec affichage classique

Important:
This script does NOT estimate an independent hidden phase variable.
It evaluates coherent tracking of the cavity mode, including
a phase-tracking diagnostic based on arg(a) and arg(a_tilde).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# =========================
# Parameters
# =========================
rng = np.random.default_rng(42)

gamma = 4.0      # damping rate
omega = 2.0      # resonance frequency
L = 1.5          # observer gain (stability: L < sqrt(gamma))
alpha = 3.0      # coherent drive amplitude
sigma = 0.05     # vacuum noise amplitude
dt = 0.002
T = 12.0

N = int(T / dt)
t = np.arange(N) * dt
s = 1 / np.sqrt(2)

A_val = -(gamma / 2 + 1j * omega)
B_val = -np.sqrt(gamma)
C_val = np.sqrt(gamma)

# Effective error eigenvalue for this discrete implementation
lam_eff = A_val + L * C_val / 2
tau = -1.0 / lam_eff.real

print(f"gamma = {gamma}")
print(f"omega = {omega}")
print(f"L = {L}")
print(f"alpha = {alpha}")
print(f"sigma = {sigma}")
print(f"A = {A_val}")
print(f"lambda_eff = {lam_eff}")
print(f"tau_conv = {tau:.3f} s")
print(f"Stability condition: Re(lambda_eff) < 0 ? {'Yes' if lam_eff.real < 0 else 'No'}")


# =========================
# Complex noise increment
# =========================
def dW():
    """Complex Wiener increment with variance dt."""
    return np.sqrt(dt / 2) * (rng.standard_normal() + 1j * rng.standard_normal())


# =========================
# Simulation
# =========================
def simulate(a0, ob0, drive=0.0):
    """
    Simulate the plant/observer coherent interconnection.

    Parameters
    ----------
    a0 : complex
        Initial plant mode.
    ob0 : complex
        Initial observer mode.
    drive : float
        Coherent drive amplitude.

    Returns
    -------
    a : ndarray
        Plant mode trajectory.
    ob : ndarray
        Observer mode trajectory.
    err : ndarray
        Squared tracking error |a - ob|^2.
    w_s : ndarray
        Innovation field trajectory.
    """
    a = np.zeros(N, dtype=complex)
    ob = np.zeros(N, dtype=complex)
    w_s = np.zeros(N, dtype=complex)

    a[0] = a0
    ob[0] = ob0

    for k in range(N - 1):
        dBin = sigma * dW()
        dB1 = sigma * dW()
        dB2 = sigma * dW()
        dB3 = sigma * dW()
        dZ = sigma * dW()

        uk = drive * np.exp(1j * omega * t[k]) * dt

        # J1: input beam splitter
        dD1 = s * (dBin + uk) + s * dB1
        dD4 = s * (dBin + uk) - s * dB1

        # output fields
        dD2 = C_val * a[k] * dt + dD1
        dD5 = C_val * ob[k] * dt + dD4

        # J2/J3: output beam splitters
        dD3 = s * (dD2 - dB2)
        dD6 = s * (dD5 - dB3)

        # J4: coherent innovation
        w = s * (dD3 - dD6)
        w_s[k] = w

        # Euler-Maruyama update
        a[k + 1] = a[k] + A_val * a[k] * dt + B_val * dD1
        ob[k + 1] = ob[k] + A_val * ob[k] * dt + B_val * dD4 - L * w - np.sqrt(gamma) * L * dZ

    w_s[-1] = w_s[-2]
    err = np.abs(a - ob) ** 2
    return a, ob, err, w_s


# =========================
# Two scenarios
# =========================
print("\nScenario A: coherent drive")
a_A, ob_A, err_A, w_A = simulate(1.5 + 0j, 0 + 0j, drive=alpha)

print("Scenario B: vacuum / no drive")
a_B, ob_B, err_B, w_B = simulate(1.0 + 0.8j, 0 + 0j, drive=0.0)

cut = int(0.7 * N)

rmse_A = np.sqrt(np.mean(err_A[cut:]))
rmse_B = np.sqrt(np.mean(err_B[cut:]))

# Phase-tracking diagnostic of the cavity mode
plant_mode_phase_A = np.angle(a_A)
observer_mode_phase_A = np.angle(ob_A)

# Wrapped phase difference for a proper mode-phase tracking error
mode_phase_diff = np.angle(np.exp(1j * (plant_mode_phase_A - observer_mode_phase_A)))
mode_phase_rmse = np.sqrt(np.mean(mode_phase_diff[cut:] ** 2))

print(f"RMSE A = {rmse_A:.6f}")
print(f"RMSE B = {rmse_B:.6f}")
print(f"Mode-phase tracking RMSE (A, t > {0.7*T:.1f}s) = {mode_phase_rmse:.6f} rad")


# =========================
# Innovation smoothing
# =========================
window = min(401, (N // 10) * 2 + 1)
if window % 2 == 0:
    window += 1
window = max(window, 5)

wA_real_s = savgol_filter(w_A.real, window_length=window, polyorder=3)
wA_imag_s = savgol_filter(w_A.imag, window_length=window, polyorder=3)


# =========================
# Plotting
# =========================
plt.style.use("default")
fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex="col")

fig.suptitle(
    "Quantum Coherent Luenberger Observer",
    fontsize=14
)

# --- 1) Real parts
ax = axes[0, 0]
ax.plot(t, a_A.real, label=r"Re$[a(t)]$ plant")
ax.plot(t, ob_A.real, "--", label=r"Re$[\tilde{a}(t)]$ observer")
ax.axvline(tau, color="k", linestyle=":", linewidth=1, label=fr"$\tau$ = {tau:.2f} s")
ax.set_title("Scenario A: Real parts")
ax.set_ylabel("Amplitude")
ax.grid(True, alpha=0.3)
ax.legend()

# --- 2) Mode phase tracking
ax = axes[0, 1]
ax.plot(t, plant_mode_phase_A, label=r"$\arg(a)$ : plant mode phase")
ax.plot(t, observer_mode_phase_A, "--", label=r"$\arg(\tilde{a})$ : observer mode phase")
ax.axvline(tau, color="k", linestyle=":", linewidth=1)
ax.set_title(f"Phase tracking of the cavity mode (RMSE = {mode_phase_rmse:.4f} rad)")
ax.set_ylabel("Phase [rad]")
ax.grid(True, alpha=0.3)
ax.legend()

# --- 3) Imaginary parts
ax = axes[1, 0]
ax.plot(t, a_A.imag, label=r"Im$[a(t)]$ plant")
ax.plot(t, ob_A.imag, "--", label=r"Im$[\tilde{a}(t)]$ observer")
ax.axvline(tau, color="k", linestyle=":", linewidth=1)
ax.set_title("Scenario A: Imaginary parts")
ax.set_ylabel("Amplitude")
ax.grid(True, alpha=0.3)
ax.legend()

# --- 4) Coherent innovation
ax = axes[1, 1]
ax.plot(t, wA_real_s, label=r"Re$[w(t)]$")
ax.plot(t, wA_imag_s, label=r"Im$[w(t)]$")
ax.axvline(tau, color="k", linestyle=":", linewidth=1)
ax.set_title("Innovation signal (smoothed)")
ax.set_ylabel("Amplitude")
ax.grid(True, alpha=0.3)
ax.legend()

# --- 5) Error energy
ax = axes[2, 0]
e2th = np.abs(a_A[0] - ob_A[0]) ** 2 * np.exp(2 * lam_eff.real * t)
ax.semilogy(t, np.maximum(err_A, 1e-14), label=r"$|a-\tilde{a}|^2$ (Scenario A)")
ax.semilogy(t, np.maximum(err_B, 1e-14), label=r"$|a-\tilde{a}|^2$ (Scenario B)")
ax.semilogy(t, np.maximum(e2th, 1e-14), "--", label="Theoretical decay")
ax.axhline(rmse_A**2, linestyle=":", label=f"Residual A = {rmse_A**2:.4e}")
ax.axhline(rmse_B**2, linestyle=":", label=f"Residual B = {rmse_B**2:.4e}")
ax.set_title("Error energy")
ax.set_xlabel("Time [s]")
ax.set_ylabel(r"$|e|^2$")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# --- 6) Phase portraits
ax = axes[2, 1]
stride = max(1, N // 1500)
ax.plot(a_A.real[::stride], a_A.imag[::stride], label="Plant (A)")
ax.plot(ob_A.real[::stride], ob_A.imag[::stride], "--", label="Observer (A)")
ax.plot(a_B.real[::stride], a_B.imag[::stride], label="Plant (B)")
ax.plot(ob_B.real[::stride], ob_B.imag[::stride], "--", label="Observer (B)")
ax.set_title("Phase portraits")
ax.set_xlabel("Re")
ax.set_ylabel("Im")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.97])

# Optional save
save_figure = True
if save_figure:
    out = os.path.join(os.getcwd(), "quantum_luenberger_classic_phase_tracking.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved -> {out}")

plt.show()