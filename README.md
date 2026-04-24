# FE5213 Project Group 4

This repository contains our FE5213 macro project notebooks and figures.
The main working file for Sections 3.2 and 3.3 is:

- `fiscal_policy_experiments_3.2_3.3.ipynb`

Section 3.1 work is in:

- `ricardian_equivalence_3.1.ipynb`

Older/inactive drafts were moved to:

- `archive/`

## Project Scope

The notebook implements:

1. A baseline RBC-style setup with endogenous labor and government spending shocks.
2. Section 3.2 fiscal shock experiments (foreseen one-time, unforeseen one-time, unforeseen permanent).
3. Section 3.3 state-dependent comparisons across productivity states `zL = 0.99` and `zH = 1.01`.
4. Diagnostics, policy maps, and plot export for report-ready figures.

## Main Outputs

Saved figures are in `plots/`:

- `fig_01pct.png`
- `fig_05pct.png`
- `fig_10pct.png`
- `fig_01pct_t120.png` (long-horizon check)
- `fig_05pct_t120.png` (long-horizon check)
- `fig_10pct_t120.png` (long-horizon check)
- `sec33_plot1.png`
- `sec33_plot2.png`
- `sec33_plot3.png`

## Dependencies

Use Python 3 with:

- `numpy`
- `scipy`
- `matplotlib`
- `pandas`

Optional:

- `quantecon` (the notebook has a fallback if unavailable for the relevant helper section).

## How to Run

1. Open `fiscal_policy_experiments_3.2_3.3.ipynb`.
2. Run cells top-to-bottom.
3. For Section 3.2 baseline figures, use the default horizon (`t=40` setup in the timing cell).
4. Run the long-horizon validation cell to generate `t=120` Section 3.2 figures:
   `fig_01pct_t120.png`, `fig_05pct_t120.png`, and `fig_10pct_t120.png`.
5. Confirm updated figures are saved in `plots/`.

## Notes

- Current Section 3.2 conditional run uses `z_sim = zH = 1.01`.
- The transition matrix used for state-dependent analysis is:

  \[
  P_z =
  \begin{bmatrix}
  0.875 & 0.125 \\
  0.125 & 0.875
  \end{bmatrix}
  \]
