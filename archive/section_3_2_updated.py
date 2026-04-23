"""
FE5213 Group Project - Section 3.2
====================================
Produces 3 figures, one per shock size (1%, 5%, 10%).

Each figure: 3 rows x 4 cols (C, L, I, Y)
  Row 1 (Foreseen one-time):    3 lines — leads 1Q, 5Q, 9Q overlaid
  Row 2 (Unforeseen one-time):  single line
  Row 3 (Unforeseen permanent): single line

Timeline (plot time):
  t=0 to t=3 : flat SS — nobody knows anything
  t=4         : FORESEEN ground zero — announcement made
  t=5         : UNFORESEEN ground zero — G hits with no warning
                also where G first hits for foreseen lead=1Q
  Foreseen leads:
    lead=1Q → G hits t=5
    lead=5Q → G hits t=9
    lead=9Q → G hits t=13

Solver domains:
  Foreseen  : solver t=0 = plot t=4, T_DYN_FOR=36 → length 37, prepend 4 → 41
  Unforeseen: solver t=0 = plot t=5, T_DYN_UNF=35 → length 36, prepend 5 → 41
  tgrid: t=0 to t=40 (41 points)

Benchmark: 10% shock, lead=5Q (G hits t=9)
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# =============================================================================
# 1. PARAMETERS
# =============================================================================

beta      = 0.988
sigma     = 1.0
phi       = 1.0
alpha     = 0.33
delta     = 0.025
z_sim     = 1.0
g_y_ratio = 0.20
target_L  = 1.0 / 3.0

T_PRE_FOR  = 4
T_DYN_FOR  = 36
T_PRE_UNF  = 5
T_DYN_UNF  = 35

GROUND_ZERO_FORESEEN   = 4
GROUND_ZERO_UNFORESEEN = 5

SHOCK_SIZES    = [0.01, 0.05, 0.10]
FORESEEN_LEADS = [1, 5, 9]
BENCHMARK_LEAD = 5
BENCHMARK_SIZE = 0.10

# =============================================================================
# 2. CORE MODEL EQUATIONS
# =============================================================================

def production_function(K, L, z):
    return z * (K ** alpha) * (L ** (1 - alpha))

def marginal_product_labor(K, L, z):
    return (1 - alpha) * z * (K ** alpha) * (L ** (-alpha))

def marginal_product_capital(K, L, z):
    return alpha * z * (K ** (alpha - 1)) * (L ** (1 - alpha))

# =============================================================================
# 3. STEADY STATE
# =============================================================================

def calibrate_chi(z=1.0, L_target=1/3, K_guess=10.0):
    def euler_eq(K_arr):
        K = K_arr[0]
        if K <= 0:
            return [1e6]
        rk = marginal_product_capital(K, L_target, z)
        return [1.0 - beta * (rk + 1.0 - delta)]

    K_ss = fsolve(euler_eq, [K_guess])[0]
    Y_ss = production_function(K_ss, L_target, z)
    I_ss = delta * K_ss
    G_ss = g_y_ratio * Y_ss
    C_ss = Y_ss - I_ss - G_ss

    if C_ss <= 0:
        raise ValueError("Calibration failed: C_ss <= 0.")

    w_ss = marginal_product_labor(K_ss, L_target, z)
    chi  = w_ss * (C_ss ** (-sigma)) / (L_target ** phi)
    return {"chi": chi, "K": K_ss, "L": L_target,
            "C": C_ss, "Y": Y_ss, "I": I_ss, "G": G_ss}


def steady_state_eqs(vars, z, chi, G_fixed):
    K, L, C = vars
    if K <= 0 or L <= 0 or C <= 0:
        return [1e6, 1e6, 1e6]
    Y  = production_function(K, L, z)
    w  = marginal_product_labor(K, L, z)
    rk = marginal_product_capital(K, L, z)
    return [1.0 - beta * (rk + 1.0 - delta),
            chi * (L ** phi) - w * (C ** (-sigma)),
            Y - C - delta * K - G_fixed]


def solve_steady_state(z, chi, G_fixed, guess=(10.0, 0.33, 1.0)):
    K, L, C = fsolve(steady_state_eqs, guess, args=(z, chi, G_fixed))
    Y = production_function(K, L, z)
    return {"K": K, "L": L, "C": C, "Y": Y, "I": delta * K,
            "G": G_fixed, "w": marginal_product_labor(K, L, z),
            "rk": marginal_product_capital(K, L, z)}

# =============================================================================
# 4. TRANSITION PATH SOLVER
# =============================================================================

def transition_system(x, K0, z, chi, G_path, terminal_K):
    T_loc       = len(G_path) - 1
    C_path      = x[0           : T_loc+1]
    L_path      = x[T_loc+1     : 2*(T_loc+1)]
    K_next_path = x[2*(T_loc+1) : 3*(T_loc+1)]
    eqs = []

    for t in range(T_loc + 1):
        K_t = K0 if t == 0 else K_next_path[t-1]
        C_t, L_t, G_t = C_path[t], L_path[t], G_path[t]
        if K_t <= 0 or C_t <= 0 or L_t <= 0:
            return np.ones(3*(T_loc+1)) * 1e6
        Y_t = production_function(K_t, L_t, z)
        w_t = marginal_product_labor(K_t, L_t, z)
        eqs.append(chi * (L_t**phi) - w_t * (C_t**(-sigma)))
        eqs.append(Y_t - C_t - G_t - (K_next_path[t] - (1-delta)*K_t))

    for t in range(T_loc):
        rk_tp1 = marginal_product_capital(K_next_path[t], L_path[t+1], z)
        eqs.append((C_path[t]**(-sigma))
                   - beta * (C_path[t+1]**(-sigma)) * (rk_tp1 + 1 - delta))

    eqs.append(K_next_path[T_loc] - terminal_K)
    return np.array(eqs)


def solve_transition(K0, z, chi, G_path, initial_ss, terminal_ss):
    T_loc = len(G_path) - 1
    x0 = np.concatenate([
        np.linspace(initial_ss["C"], terminal_ss["C"], T_loc+1),
        np.linspace(initial_ss["L"], terminal_ss["L"], T_loc+1),
        np.linspace(initial_ss["K"], terminal_ss["K"], T_loc+1),
    ])
    sol = fsolve(transition_system, x0,
                 args=(K0, z, chi, G_path, terminal_ss["K"]),
                 xtol=1e-10, maxfev=50000)

    C_path      = sol[0          : T_loc+1]
    L_path      = sol[T_loc+1    : 2*(T_loc+1)]
    K_next_path = sol[2*(T_loc+1): 3*(T_loc+1)]

    K_path    = np.zeros(T_loc+1)
    K_path[0] = K0
    for t in range(1, T_loc+1):
        K_path[t] = K_next_path[t-1]

    Y_path = np.array([production_function(K_path[t], L_path[t], z)
                       for t in range(T_loc+1)])
    I_path = np.zeros(T_loc+1)
    for t in range(T_loc):
        I_path[t] = K_path[t+1] - (1-delta)*K_path[t]
    I_path[T_loc] = K_next_path[T_loc] - (1-delta)*K_path[T_loc]

    return {"K": K_path, "C": C_path, "L": L_path,
            "Y": Y_path, "I": I_path, "G": G_path}


def prepend_ss(res, ss, n_pre):
    out = {}
    for var in ["C", "L", "Y", "I", "K", "G"]:
        pre = np.full(n_pre, ss[var])
        out[var] = np.concatenate([pre, res[var]])
    return out

# =============================================================================
# 5. SHOCK PATH BUILDERS
# =============================================================================

def make_foreseen_path(G_ss, shock_size, lead):
    path = np.full(T_DYN_FOR + 1, G_ss)
    if lead <= T_DYN_FOR:
        path[lead] = G_ss * (1.0 + shock_size)
    return path

def make_unforeseen_onetime_path(G_ss, shock_size):
    path = np.full(T_DYN_UNF + 1, G_ss)
    path[0] = G_ss * (1.0 + shock_size)
    return path

def make_permanent_path(G_ss, shock_size):
    return np.full(T_DYN_UNF + 1, G_ss * (1.0 + shock_size))

# =============================================================================
# 6. PLOTTING
# =============================================================================

def pct_dev(path, ss_val):
    return 100.0 * (path - ss_val) / ss_val


def plot_figure(foreseen_by_lead, res_unforeseen, res_permanent,
                ss, shock_size_pct, save_path=None):

    vars_plot  = [("C", "Consumption"), ("L", "Labour Supply"),
                  ("I", "Investment"),  ("Y", "Output")]
    row_labels = ["(1) Foreseen one-time",
                  "(2) Unforeseen one-time",
                  "(3) Unforeseen permanent"]
    colors     = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    linestyles = ["-", "--", ":"]

    tgrid = np.arange(T_PRE_FOR + T_DYN_FOR + 1)   # 41 points t=0..40

    fig, axes = plt.subplots(
        3, 4, figsize=(15, 9),
        gridspec_kw={"hspace": 0.65, "wspace": 0.38}
    )

    is_benchmark = (shock_size_pct == int(BENCHMARK_SIZE * 100))

    for col, (var, var_label) in enumerate(vars_plot):

        # ── Row 0: foreseen ───────────────────────────────────────────────────
        ax = axes[0, col]
        for lead, color, ls in zip(FORESEEN_LEADS, colors, linestyles):
            lw    = 2.4 if (lead == BENCHMARK_LEAD and is_benchmark) else 1.6
            label = (f"lead={lead}Q"
                     if lead == BENCHMARK_LEAD and is_benchmark
                     else f"lead={lead}Q")
            full = prepend_ss(foreseen_by_lead[lead], ss, n_pre=T_PRE_FOR)
            ax.plot(tgrid, pct_dev(full[var], ss[var]),
                    label=label, color=color, linewidth=lw, linestyle=ls)

        ax.axvline(GROUND_ZERO_FORESEEN, color="gray", linewidth=0.8,
                   linestyle="--", alpha=0.7)
        ax.text(GROUND_ZERO_FORESEEN, 1, f"t={GROUND_ZERO_FORESEEN} (announced)",
                fontsize=6, color="gray", fontweight="bold",
                ha="center", va="bottom", transform=ax.get_xaxis_transform())
        ax.axvline(GROUND_ZERO_UNFORESEEN, color="dimgray", linewidth=0.8,
                   linestyle="--", alpha=0.7, ymin=0, ymax=1)
        ax.text(GROUND_ZERO_UNFORESEEN, -0.03, f"t={GROUND_ZERO_UNFORESEEN}",
                fontsize=6, color="dimgray", fontweight="bold",
                ha="center", va="top", transform=ax.get_xaxis_transform())
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(var_label, fontsize=10, fontweight="bold", pad=6)
        ax.set_ylabel(row_labels[0] + "\n% dev. from SS"
                      if col == 0 else "% dev. from SS", fontsize=7.5)
        ax.set_xlabel("t", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=7, title="Foreseen lead",
                  title_fontsize=7, loc="best")

        # ── Row 1: unforeseen one-time ─────────────────────────────────────────
        ax = axes[1, col]
        full = prepend_ss(res_unforeseen, ss, n_pre=T_PRE_UNF)
        ax.plot(tgrid, pct_dev(full[var], ss[var]),
                color="#1f77b4", linewidth=1.6)
        ax.axvline(GROUND_ZERO_UNFORESEEN, color="dimgray", linewidth=0.8,
                   linestyle="--", alpha=0.7, ymin=0, ymax=1)
        ax.text(GROUND_ZERO_UNFORESEEN, -0.03, f"t={GROUND_ZERO_UNFORESEEN}",
                fontsize=6, color="dimgray", fontweight="bold",
                ha="center", va="top", transform=ax.get_xaxis_transform())
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_ylabel(row_labels[1] + "\n% dev. from SS"
                      if col == 0 else "% dev. from SS", fontsize=7.5)
        ax.set_xlabel("t", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # ── Row 2: unforeseen permanent ────────────────────────────────────────
        ax = axes[2, col]
        full = prepend_ss(res_permanent, ss, n_pre=T_PRE_UNF)
        ax.plot(tgrid, pct_dev(full[var], ss[var]),
                color="#1f77b4", linewidth=1.6)
        ax.axvline(GROUND_ZERO_UNFORESEEN, color="dimgray", linewidth=0.8,
                   linestyle="--", alpha=0.7, ymin=0, ymax=1)
        ax.text(GROUND_ZERO_UNFORESEEN, -0.03, f"t={GROUND_ZERO_UNFORESEEN} ",
                fontsize=6, color="dimgray", fontweight="bold",
                ha="center", va="top", transform=ax.get_xaxis_transform())
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_ylabel(row_labels[2] + "\n% dev. from SS"
                      if col == 0 else "% dev. from SS", fontsize=7.5)
        ax.set_xlabel("t", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Single legend for dashed lines in the figure suptitle area
    benchmark_note = "  [Benchmark: 10% shock, lead=5Q]" if is_benchmark else ""
    fig.suptitle(
        f"Section 3.2: Government Spending Shock IRFs  —  "
        f"Shock size = {shock_size_pct}% of G_ss{benchmark_note}\n",
        fontsize=10, fontweight="bold", y=1.02
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close(fig)

# =============================================================================
# 7. MAIN
# =============================================================================

if __name__ == "__main__":

    calib  = calibrate_chi(z=z_sim, L_target=target_L)
    chi    = calib["chi"]
    ss_ref = solve_steady_state(z_sim, chi, calib["G"])

    print("=" * 55)
    print("BASELINE STEADY STATE")
    print("=" * 55)
    for k, v in ss_ref.items():
        print(f"  {k:<6}: {v:.4f}")
    print(f"  {'chi':<6}: {chi:.4f}")
    print(f"\nForeseen  : announced t={GROUND_ZERO_FORESEEN}, "
          f"G hits t={GROUND_ZERO_FORESEEN+1}/"
          f"t={GROUND_ZERO_FORESEEN+5}/"
          f"t={GROUND_ZERO_FORESEEN+9}")
    print(f"Unforeseen: G hits t={GROUND_ZERO_UNFORESEEN} (no prior notice)")
    print(f"Benchmark : {int(BENCHMARK_SIZE*100)}% shock, lead={BENCHMARK_LEAD}Q "
          f"(G hits t={GROUND_ZERO_FORESEEN+BENCHMARK_LEAD})\n")

    G_ss = ss_ref["G"]

    for shock_size in SHOCK_SIZES:
        size_pct = int(shock_size * 100)
        print(f"\n--- Shock size = {size_pct}% ---")

        foreseen_by_lead = {}
        for lead in FORESEEN_LEADS:
            G_path = make_foreseen_path(G_ss, shock_size, lead)
            foreseen_by_lead[lead] = solve_transition(
                ss_ref["K"], z_sim, chi, G_path, ss_ref, ss_ref)
            print(f"  Foreseen lead={lead}Q "
                  f"(announced t={GROUND_ZERO_FORESEEN}, "
                  f"G hits t={GROUND_ZERO_FORESEEN+lead}) done.")

        G_unf = make_unforeseen_onetime_path(G_ss, shock_size)
        res_unforeseen = solve_transition(
            ss_ref["K"], z_sim, chi, G_unf, ss_ref, ss_ref)
        print(f"  Unforeseen one-time (G hits t={GROUND_ZERO_UNFORESEEN}) done.")

        terminal_perm = solve_steady_state(
            z_sim, chi, G_ss * (1.0 + shock_size),
            guess=(ss_ref["K"], ss_ref["L"], ss_ref["C"])
        )
        G_perm = make_permanent_path(G_ss, shock_size)
        res_permanent = solve_transition(
            ss_ref["K"], z_sim, chi, G_perm, ss_ref, terminal_perm)
        print(f"  Unforeseen permanent (G rises t={GROUND_ZERO_UNFORESEEN}) done.")

        plot_figure(
            foreseen_by_lead, res_unforeseen, res_permanent,
            ss_ref, size_pct,
            save_path=f"fig_{size_pct:02d}pct.png"
        )

    print("\nDone. 3 figures saved: fig_01pct.png, fig_05pct.png, fig_10pct.png")