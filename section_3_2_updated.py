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
T         = 40

SHOCK_SIZES    = [0.01, 0.05, 0.10]   # 1%, 5%, 10% of G_ss
FORESEEN_DATES = [1, 4, 8]            # announcement leads 

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
    return {"K": K, "L": L, "C": C, "Y": Y, "I": delta*K,
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
        eqs.append(chi*(L_t**phi) - w_t*(C_t**(-sigma)))
        eqs.append(Y_t - C_t - G_t - (K_next_path[t] - (1-delta)*K_t))

    for t in range(T_loc):
        rk_tp1 = marginal_product_capital(K_next_path[t], L_path[t+1], z)
        eqs.append((C_path[t]**(-sigma)) - beta*(C_path[t+1]**(-sigma))*(rk_tp1+1-delta))

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

    C_path      = sol[0       : T_loc+1]
    L_path      = sol[T_loc+1 : 2*(T_loc+1)]
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

# =============================================================================
# 5. SHOCK PATH BUILDERS
# =============================================================================

def make_foreseen_path(G_ss, shock_size, T_sim, lead):
    """Announced at t=0, hits once at t=lead."""
    path = np.full(T_sim+1, G_ss)
    path[lead] = G_ss * (1.0 + shock_size)
    return path

def make_unforeseen_onetime_path(G_ss, shock_size, T_sim):
    """No advance notice, hits once at t=0."""
    path = np.full(T_sim+1, G_ss)
    path[0] = G_ss * (1.0 + shock_size)
    return path

def make_permanent_path(G_ss, shock_size, T_sim):
    """No advance notice, permanent increase from t=0."""
    return np.full(T_sim+1, G_ss * (1.0 + shock_size))

# =============================================================================
# 6. PLOTTING
# =============================================================================

def pct_dev(path, ss_val):
    return 100.0 * (path - ss_val) / ss_val


def plot_figure(foreseen_by_lead, res_unforeseen, res_permanent,
                ss, shock_size_pct, save_path=None):
    """
    One figure per shock size. 3 rows x 4 cols.
      Row 1: foreseen — 3 coloured lines (lead 1Q, 4Q, 8Q overlaid)
      Row 2: unforeseen one-time — single line
      Row 3: unforeseen permanent — single line
    """
    vars_plot  = [("C","Consumption"), ("L","Labour Supply"),
                  ("I","Investment"),  ("Y","Output")]
    row_labels = [
        "(1) Foreseen one-time",
        "(2) Unforeseen one-time",
        "(3) Unforeseen permanent",
    ]
    colors     = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    linestyles = ["-", "--", ":"]
    tgrid      = np.arange(T + 1)

    fig, axes = plt.subplots(
        3, 4, figsize=(15, 9),
        gridspec_kw={"hspace": 0.55, "wspace": 0.35}
    )

    for col, (var, var_label) in enumerate(vars_plot):

        # Row 0: foreseen, 3 leads overlaid
        ax = axes[0, col]
        for (lead, color, ls) in zip(FORESEEN_DATES, colors, linestyles):
            ax.plot(tgrid, pct_dev(foreseen_by_lead[lead][var], ss[var]),
                    label=f"t={lead}", color=color,
                    linewidth=1.6, linestyle=ls)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(var_label, fontsize=10, fontweight="bold", pad=6)
        if col == 0:
            ax.set_ylabel(row_labels[0] + "\n% dev. from SS", fontsize=7.5)
        ax.set_xlabel("t", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=7, title="Shock hits at",
                  title_fontsize=7, loc="best")

        # Row 1: unforeseen one-time
        ax = axes[1, col]
        ax.plot(tgrid, pct_dev(res_unforeseen[var], ss[var]),
                color="#1f77b4", linewidth=1.6)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        if col == 0:
            ax.set_ylabel(row_labels[1] + "\n% dev. from SS", fontsize=7.5)
        ax.set_xlabel("t", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Row 2: unforeseen permanent
        ax = axes[2, col]
        ax.plot(tgrid, pct_dev(res_permanent[var], ss[var]),
                color="#1f77b4", linewidth=1.6)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        if col == 0:
            ax.set_ylabel(row_labels[2] + "\n% dev. from SS", fontsize=7.5)
        ax.set_xlabel("t", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Section 3.2: Government Spending Shock IRFs  —  Shock size = {shock_size_pct}% of G_ss",
        fontsize=12, fontweight="bold", y=1.01
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

    print("=" * 50)
    print("BASELINE STEADY STATE")
    print("=" * 50)
    for k, v in ss_ref.items():
        print(f"  {k:<6}: {v:.4f}")
    print(f"  {'chi':<6}: {chi:.4f}\n")

    G_ss = ss_ref["G"]

    for shock_size in SHOCK_SIZES:
        size_pct = int(shock_size * 100)
        print(f"\n--- Shock size = {size_pct}% ---")

        # Foreseen: 3 leads (timing is meaningful here)
        foreseen_by_lead = {}
        for lead in FORESEEN_DATES:
            G_path = make_foreseen_path(G_ss, shock_size, T, lead)
            foreseen_by_lead[lead] = solve_transition(
                ss_ref["K"], z_sim, chi, G_path, ss_ref, ss_ref)
            print(f"  Foreseen lead={lead}Q done.")

        # Unforeseen one-time: solved once at t=0
        G_unf = make_unforeseen_onetime_path(G_ss, shock_size, T)
        res_unforeseen = solve_transition(
            ss_ref["K"], z_sim, chi, G_unf, ss_ref, ss_ref)
        print(f"  Unforeseen one-time done.")

        # Unforeseen permanent: solved once at t=0
        terminal_perm = solve_steady_state(
            z_sim, chi, G_ss * (1.0 + shock_size),
            guess=(ss_ref["K"], ss_ref["L"], ss_ref["C"])
        )
        G_perm = make_permanent_path(G_ss, shock_size, T)
        res_permanent = solve_transition(
            ss_ref["K"], z_sim, chi, G_perm, ss_ref, terminal_perm)
        print(f"  Unforeseen permanent done.")

        plot_figure(
            foreseen_by_lead, res_unforeseen, res_permanent,
            ss_ref, size_pct,
            save_path=f"fig_{size_pct:02d}pct.png"
        )

    print("\nDone. 3 figures saved: fig_01pct.png, fig_05pct.png, fig_10pct.png")