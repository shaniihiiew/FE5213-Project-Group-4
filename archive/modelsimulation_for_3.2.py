import numpy as np
from scipy.optimize import fsolve

# 1. Parameters

# Household
beta  = 0.99
sigma = 2.0
phi   = 1.0

# Firm
alpha = 0.33
delta = 0.025

# Productivity states
z_states = np.array([1.02, 0.98])   # [z_H, z_L]

# Government
g_y_ratio = 0.2
initial_B = 0.0

# Calibration target
target_L = 1.0 / 3.0

# 2. Core model equations

def production_function(K, L, z):
    return z * (K ** alpha) * (L ** (1 - alpha))

def marginal_product_labor(K, L, z):
    return (1 - alpha) * z * (K ** alpha) * (L ** (-alpha))

def marginal_product_capital(K, L, z):
    return alpha * z * (K ** (alpha - 1)) * (L ** (1 - alpha))

def household_utility(C, L, chi):
    if C <= 0 or L < 0:
        return -np.inf

    if sigma == 1:
        cons_utility = np.log(C)
    else:
        cons_utility = (C ** (1 - sigma) - 1) / (1 - sigma)

    labor_cost = chi * (L ** (1 + phi)) / (1 + phi)
    return cons_utility - labor_cost

def resource_constraint(C, I, G, Y):
    return Y - (C + I + G)


# 3. Calibrate chi to hit target steady-state labor

def calibrate_chi(z=1.0, L_target=1/3, K_guess=10.0):
    """
    Fix L = L_target.
    Solve Euler equation for K.
    Then back out chi from:

        chi * L^phi = w * C^(-sigma)

    => chi = w * C^(-sigma) / L^phi
    """

    def euler_equation_for_K(K_array):
        K = K_array[0]

        if K <= 0:
            return [1e6]

        rk = marginal_product_capital(K, L_target, z)
        return [1.0 - beta * (rk + 1.0 - delta)]

    K_ss = fsolve(euler_equation_for_K, [K_guess])[0]

    Y_ss = production_function(K_ss, L_target, z)
    I_ss = delta * K_ss
    G_ss = g_y_ratio * Y_ss
    C_ss = Y_ss - I_ss - G_ss

    if C_ss <= 0:
        raise ValueError("Calibration failed: steady-state consumption is non-positive.")

    w_ss = marginal_product_labor(K_ss, L_target, z)

    chi_value = w_ss * (C_ss ** (-sigma)) / (L_target ** phi)

    return {
        "chi": chi_value,
        "K": K_ss,
        "L": L_target,
        "Y": Y_ss,
        "C": C_ss,
        "I": I_ss,
        "G": G_ss,
        "w": w_ss,
        "rk": marginal_product_capital(K_ss, L_target, z)
    }


# 4. Solve steady state given chi

def steady_state_equations(variables, z, chi):
    K, L, C = variables

    if K <= 0 or L <= 0 or C <= 0:
        return [1e6, 1e6, 1e6]

    Y = production_function(K, L, z)
    w = marginal_product_labor(K, L, z)
    rk = marginal_product_capital(K, L, z)

    G = g_y_ratio * Y
    I = delta * K

    # Euler equation
    error1 = 1.0 - beta * (rk + 1.0 - delta)

    # Labor FOC
    error2 = chi * (L ** phi) - w * (C ** (-sigma))

    # Resource constraint
    error3 = Y - C - I - G

    return [error1, error2, error3]


def solve_steady_state(z, chi, initial_guess=(10.0, 0.33, 1.0)):
    sol = fsolve(steady_state_equations, initial_guess, args=(z, chi))
    K_ss, L_ss, C_ss = sol

    Y_ss = production_function(K_ss, L_ss, z)
    I_ss = delta * K_ss
    G_ss = g_y_ratio * Y_ss

    return {
        "K": K_ss,
        "L": L_ss,
        "C": C_ss,
        "Y": Y_ss,
        "I": I_ss,
        "G": G_ss
    }


# 5. Main

if __name__ == "__main__":
    calibration = calibrate_chi(z=1.0, L_target=target_L, K_guess=10.0)
    chi = calibration["chi"]

    ss = solve_steady_state(
        z=1.0,
        chi=chi,
        initial_guess=(calibration["K"], target_L, calibration["C"])
    )

    print("Steady State Results:")
    print(f"Labor (L): {ss['L']:.4f}")
    print(f"Capital (K): {ss['K']:.4f}")
    print(f"Consumption (C): {ss['C']:.4f}")
    print(f"Output (Y): {ss['Y']:.4f}")

# Section 3.2 
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# 1. Parameters

# Household
beta  = 0.99
sigma = 2.0
phi   = 1.0

# Firm
alpha = 0.33
delta = 0.025

# Productivity state used for Section 3.2
z_sim = 1.0

# Government
g_y_ratio = 0.2          # baseline steady-state G/Y ratio
target_L  = 1.0 / 3.0    # calibrate chi so that L_ss = 1/3

# Shock settings
shock_size = 0.10        # 10% government spending shock
T = 40                   # simulation horizon
foreseen_shock_date = 5  # for scenario (1)


# 2. Core model equations

def production_function(K, L, z):
    return z * (K ** alpha) * (L ** (1 - alpha))

def marginal_product_labor(K, L, z):
    return (1 - alpha) * z * (K ** alpha) * (L ** (-alpha))

def marginal_product_capital(K, L, z):
    return alpha * z * (K ** (alpha - 1)) * (L ** (1 - alpha))


# 3. Calibrate chi so that steady-state labor = 1/3

def calibrate_chi(z=1.0, L_target=1/3, K_guess=10.0):
    """
    Baseline calibration uses:
        G_ss = g_y_ratio * Y_ss

    Fix L = L_target, solve Euler equation for K,
    then back out chi from the labor FOC:
        chi * L^phi = w * C^(-sigma)
    """

    def euler_equation_for_K(K_array):
        K = K_array[0]

        if K <= 0:
            return [1e6]

        rk = marginal_product_capital(K, L_target, z)
        return [1.0 - beta * (rk + 1.0 - delta)]

    K_ss = fsolve(euler_equation_for_K, [K_guess])[0]

    Y_ss = production_function(K_ss, L_target, z)
    I_ss = delta * K_ss
    G_ss = g_y_ratio * Y_ss
    C_ss = Y_ss - I_ss - G_ss

    if C_ss <= 0:
        raise ValueError("Calibration failed: steady-state consumption is non-positive.")

    w_ss = marginal_product_labor(K_ss, L_target, z)
    chi_value = w_ss * (C_ss ** (-sigma)) / (L_target ** phi)

    return {
        "chi": chi_value,
        "K": K_ss,
        "L": L_target,
        "C": C_ss,
        "Y": Y_ss,
        "I": I_ss,
        "G": G_ss
    }

# 4. Solve steady state for a given fixed level of G

def steady_state_fixed_G_equations(vars, z, chi, G_fixed):
    """
    Unknowns: K, L, C

    Steady-state conditions:
        1 = beta * (rk + 1 - delta)
        chi * L^phi = w * C^(-sigma)
        Y = C + delta*K + G_fixed
    """
    K, L, C = vars

    if K <= 0 or L <= 0 or C <= 0:
        return [1e6, 1e6, 1e6]

    Y  = production_function(K, L, z)
    w  = marginal_product_labor(K, L, z)
    rk = marginal_product_capital(K, L, z)

    eq1 = 1.0 - beta * (rk + 1.0 - delta)
    eq2 = chi * (L ** phi) - w * (C ** (-sigma))
    eq3 = Y - C - delta * K - G_fixed

    return [eq1, eq2, eq3]


def solve_steady_state_fixed_G(z, chi, G_fixed, initial_guess=(10.0, 0.33, 1.0)):
    sol = fsolve(
        steady_state_fixed_G_equations,
        initial_guess,
        args=(z, chi, G_fixed)
    )

    K_ss, L_ss, C_ss = sol
    Y_ss  = production_function(K_ss, L_ss, z)
    I_ss  = delta * K_ss
    w_ss  = marginal_product_labor(K_ss, L_ss, z)
    rk_ss = marginal_product_capital(K_ss, L_ss, z)

    return {
        "K": K_ss,
        "L": L_ss,
        "C": C_ss,
        "Y": Y_ss,
        "I": I_ss,
        "G": G_fixed,
        "w": w_ss,
        "rk": rk_ss
    }

# 5. Transition path solver (perfect foresight)

def pack_transition_variables(C_path, L_path, K_next_path):
    return np.concatenate([C_path, L_path, K_next_path])

def unpack_transition_variables(x, T):
    C_path      = x[0:T+1]
    L_path      = x[T+1:2*(T+1)]
    K_next_path = x[2*(T+1):3*(T+1)]
    return C_path, L_path, K_next_path


def transition_system(x, K0, z, chi, G_path, terminal_K):
    """
    Unknowns:
        C_t, L_t, K_{t+1}   for t = 0,...,T

    Equations:
        (i)   labor FOC, t=0,...,T
        (ii)  resource/capital accumulation, t=0,...,T
        (iii) Euler equation, t=0,...,T-1
        (iv)  terminal condition K_{T+1} = terminal_K
    """
    T = len(G_path) - 1
    C_path, L_path, K_next_path = unpack_transition_variables(x, T)

    eqs = []

    # 1. Labor FOC and resource constraint for each t
    for t in range(T + 1):
        K_t = K0 if t == 0 else K_next_path[t - 1]
        C_t = C_path[t]
        L_t = L_path[t]
        G_t = G_path[t]

        if K_t <= 0 or C_t <= 0 or L_t <= 0:
            return np.ones(3 * (T + 1)) * 1e6

        Y_t = production_function(K_t, L_t, z)
        w_t = marginal_product_labor(K_t, L_t, z)

        # labor FOC
        eq_labor = chi * (L_t ** phi) - w_t * (C_t ** (-sigma))
        eqs.append(eq_labor)

        # resource constraint + capital accumulation:
        # Y_t = C_t + G_t + K_{t+1} - (1-delta)K_t
        eq_resource = Y_t - C_t - G_t - (K_next_path[t] - (1.0 - delta) * K_t)
        eqs.append(eq_resource)

    # 2. Euler equations for t = 0,...,T-1
    for t in range(T):
        K_tp1 = K_next_path[t]
        C_t   = C_path[t]
        C_tp1 = C_path[t + 1]
        L_tp1 = L_path[t + 1]

        rk_tp1 = marginal_product_capital(K_tp1, L_tp1, z)

        eq_euler = (C_t ** (-sigma)) - beta * (C_tp1 ** (-sigma)) * (rk_tp1 + 1.0 - delta)
        eqs.append(eq_euler)

    # 3. Terminal condition on capital
    eq_terminal = K_next_path[T] - terminal_K
    eqs.append(eq_terminal)

    return np.array(eqs)


def solve_transition_path(K0, z, chi, G_path, initial_ss, terminal_ss):
    T = len(G_path) - 1

    # Initial guesses
    C_guess = np.linspace(initial_ss["C"], terminal_ss["C"], T + 1)
    L_guess = np.linspace(initial_ss["L"], terminal_ss["L"], T + 1)
    K_guess = np.linspace(initial_ss["K"], terminal_ss["K"], T + 1)

    x0 = pack_transition_variables(C_guess, L_guess, K_guess)

    sol = fsolve(
        transition_system,
        x0,
        args=(K0, z, chi, G_path, terminal_ss["K"]),
        xtol=1e-10,
        maxfev=50000
    )

    C_path, L_path, K_next_path = unpack_transition_variables(sol, T)

    # Build K_t path including initial K0
    K_path = np.zeros(T + 1)
    K_path[0] = K0
    for t in range(1, T + 1):
        K_path[t] = K_next_path[t - 1]

    Y_path = np.zeros(T + 1)
    I_path = np.zeros(T + 1)

    for t in range(T + 1):
        Y_path[t] = production_function(K_path[t], L_path[t], z)
        if t < T:
            I_path[t] = K_path[t + 1] - (1.0 - delta) * K_path[t]
        else:
            I_path[t] = K_next_path[T] - (1.0 - delta) * K_path[t]

    return {
        "K": K_path,
        "C": C_path,
        "L": L_path,
        "Y": Y_path,
        "I": I_path,
        "G": G_path
    }


# 6. Government spending paths for Section 3.2

def make_foreseen_one_time_G_path(G_ss, shock_size, T, shock_date):
    G_path = np.full(T + 1, G_ss)
    G_path[shock_date] = G_ss * (1.0 + shock_size)
    return G_path

def make_unforeseen_one_time_G_path(G_ss, shock_size, T):
    G_path = np.full(T + 1, G_ss)
    G_path[0] = G_ss * (1.0 + shock_size)
    return G_path

def make_permanent_G_path(G_ss, shock_size, T):
    G_path = np.full(T + 1, G_ss * (1.0 + shock_size))
    return G_path


# 7. Plotting

def percent_deviation(path, steady_value):
    return 100.0 * (path - steady_value) / steady_value

def plot_irfs(results, baseline_ss, title):
    tgrid = np.arange(len(results["C"]))

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()

    series = [
        ("Consumption", results["C"], baseline_ss["C"]),
        ("Labor",       results["L"], baseline_ss["L"]),
        ("Investment",  results["I"], baseline_ss["I"]),
        ("Output",      results["Y"], baseline_ss["Y"]),
        ("Capital",     results["K"], baseline_ss["K"]),
        ("Government",  results["G"], baseline_ss["G"]),
    ]

    for ax, (name, path, ss_val) in zip(axes, series):
        ax.plot(tgrid, percent_deviation(path, ss_val))
        ax.axhline(0.0, linewidth=0.8)
        ax.set_title(name)
        ax.set_xlabel("Time")
        ax.set_ylabel("% dev. from SS")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    plt.show()


# 8. Main: Section 3.2

if __name__ == "__main__":

    # Step A: calibrate chi and solve baseline steady state
    calibration = calibrate_chi(z=z_sim, L_target=target_L, K_guess=10.0)
    chi = calibration["chi"]

    baseline_ss = solve_steady_state_fixed_G(
        z=z_sim,
        chi=chi,
        G_fixed=calibration["G"],
        initial_guess=(calibration["K"], calibration["L"], calibration["C"])
    )

    print("Baseline Steady State:")
    print(f"Labor (L): {baseline_ss['L']:.4f}")
    print(f"Capital (K): {baseline_ss['K']:.4f}")
    print(f"Consumption (C): {baseline_ss['C']:.4f}")
    print(f"Output (Y): {baseline_ss['Y']:.4f}")
    print(f"Government Spending (G): {baseline_ss['G']:.4f}")
    print()

    # Step B: build the three G paths

    # (1) Foreseen one-time shock at t = foreseen_shock_date
    G_path_foreseen = make_foreseen_one_time_G_path(
        G_ss=baseline_ss["G"],
        shock_size=shock_size,
        T=T,
        shock_date=foreseen_shock_date
    )

    # (2) Unforeseen one-time shock at t = 0
    G_path_unforeseen = make_unforeseen_one_time_G_path(
        G_ss=baseline_ss["G"],
        shock_size=shock_size,
        T=T
    )

    # (3) Unforeseen permanent shock from t = 0 onward
    G_path_permanent = make_permanent_G_path(
        G_ss=baseline_ss["G"],
        shock_size=shock_size,
        T=T
    )

    # Step C: terminal steady states

    # temporary shocks return to original steady state
    terminal_ss_temporary = baseline_ss

    # permanent shock converges to a new steady state with higher G
    G_perm = baseline_ss["G"] * (1.0 + shock_size)
    permanent_ss = solve_steady_state_fixed_G(
        z=z_sim,
        chi=chi,
        G_fixed=G_perm,
        initial_guess=(baseline_ss["K"], baseline_ss["L"], baseline_ss["C"])
    )

    # Step D: solve transition paths

    results_foreseen = solve_transition_path(
        K0=baseline_ss["K"],
        z=z_sim,
        chi=chi,
        G_path=G_path_foreseen,
        initial_ss=baseline_ss,
        terminal_ss=terminal_ss_temporary
    )

    results_unforeseen = solve_transition_path(
        K0=baseline_ss["K"],
        z=z_sim,
        chi=chi,
        G_path=G_path_unforeseen,
        initial_ss=baseline_ss,
        terminal_ss=terminal_ss_temporary
    )

    results_permanent = solve_transition_path(
        K0=baseline_ss["K"],
        z=z_sim,
        chi=chi,
        G_path=G_path_permanent,
        initial_ss=baseline_ss,
        terminal_ss=permanent_ss
    )

    # Step E: plot IRFs
    plot_irfs(results_foreseen, baseline_ss, "Section 3.2: Foreseen One-Time Government Spending Shock")
    plot_irfs(results_unforeseen, baseline_ss, "Section 3.2: Unforeseen One-Time Government Spending Shock")
    plot_irfs(results_permanent, baseline_ss, "Section 3.2: Unforeseen Permanent Government Spending Shock")