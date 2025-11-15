# -*- coding: utf-8 -*-
Analysis of the composite endpoint under various priors in Python through Colab
Samuel Heuts
November 15th, 2025

# ============================================
# SLIM COMPOSITE ENDPOINT — FINAL PRIOR ANALYSES
# Composite: all-cause death, MI, any revascularization, stroke
# Data: 13/240 (complete) vs 32/238 (culprit-only)
# ============================================

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde

# --------------------------------------------
# Common data for the composite endpoint
# --------------------------------------------
a  = 13   # events in complete group
n1 = 240
c  = 32   # events in culprit-only group
n0 = 238

p_c = c / n0   # baseline risk in culprit-only
logRR_hat = math.log((a / n1) / (c / n0))
se_logRR  = math.sqrt(1/a - 1/n1 + 1/c - 1/n0)

print("COMPOSITE ENDPOINT (Primary)")
print(f"Events: complete = {a}/{n1}, culprit-only = {c}/{n0}")
print(f"Control risk p_c = {p_c:.4f} ({p_c*100:.2f}%)")
print(f"logRR (MLE)     = {logRR_hat:.3f}")
print(f"SE(logRR)       = {se_logRR:.3f}\n")


# --------------------------------------------
# Helper: build priors on log RR
# --------------------------------------------

# MCID: -5% absolute risk difference (ARD = -0.05) in favor of complete
delta_mcid = -0.05
theta_mcid = math.log(1 + delta_mcid / p_c)   # solution of p_c*(exp(theta)-1) = -0.05

# Skeptical prior:
# logRR ~ Normal(0, sigma^2) with P(ARD <= -5%) = 0.10
# ARD <= -5% <=> logRR <= theta_mcid (monotone)
# => P(logRR <= theta_mcid) = 0.10 => theta_mcid = z_0.10 * sigma
z_0_10 = norm.ppf(0.10)
sigma_skeptical = theta_mcid / z_0_10   # negative / negative -> positive

# Enthusiastic prior:
# Centered at MCID (theta_mcid), with 30% probability of harm (logRR > 0)
# => P(logRR > 0) = 0.30 => 1 - Phi((0 - mu)/sigma) = 0.30
# => Phi(-mu/sigma) = 0.70 => -mu/sigma = z_0.70
z_0_70 = norm.ppf(0.70)
sigma_enth = -theta_mcid / z_0_70     # mu = theta_mcid < 0

# Pessimistic prior: mirror of enthusiastic
# mean = -theta_mcid, sd = sigma_enth
sigma_pess = sigma_enth

# Literature-based prior (NSTEMI IPD meta-analysis: 68/536 vs 91/532)
a_meta, n1_meta = 68, 536
c_meta, n0_meta = 91, 532
logRR_meta = math.log((a_meta/n1_meta) / (c_meta/n0_meta))
se_logRR_meta = math.sqrt(1/a_meta - 1/n1_meta + 1/c_meta - 1/n0_meta)


# --------------------------------------------
# Define priors: (name, mean_logRR, sd_logRR)
# --------------------------------------------
priors = [
    ("Weakly informative",
     0.0,
     2.0),

    ("Skeptical (10% at MCID benefit)",
     0.0,
     float(sigma_skeptical)),

    ("Enthusiastic (centered at MCID, 30% harm)",
     float(theta_mcid),
     float(sigma_enth)),

    ("Pessimistic (centered at MCID harm, 30% benefit)",
     -float(theta_mcid),
     float(sigma_pess)),

    ("Literature-based (NSTEMI IPD meta-analysis)",
     float(logRR_meta),
     float(se_logRR_meta)),
]


# --------------------------------------------
# Helper function to run full analysis under a given prior
# --------------------------------------------
def analyze_composite_prior(prior_name, mu0, sd0,
                            n_draws=200000,
                            seed=123,
                            x_min=-15.0, x_max=5.0):
    """
    Normal likelihood on logRR + Normal prior on logRR.
    Posterior Normal on logRR, sampling-based summaries on RR and ARD.
    """
    print("\n" + "="*70)
    print(prior_name.upper())
    print("="*70)

    # Prior summary
    print(f"Prior (logRR) ~ Normal(mean={mu0:.3f}, sd={sd0:.3f})")

    # Posterior on logRR (Normal-Normal conjugacy, using Wald likelihood)
    v_prior = sd0**2
    v_like  = se_logRR**2

    v_post = 1.0 / (1.0/v_prior + 1.0/v_like)
    m_post = v_post * (mu0/v_prior + logRR_hat/v_like)
    sd_post = math.sqrt(v_post)

    print(f"Posterior (logRR) ~ Normal(mean={m_post:.3f}, sd={sd_post:.3f})")

    # Draw samples from posterior on logRR
    rng = np.random.default_rng(seed)
    theta = rng.normal(loc=m_post, scale=sd_post, size=n_draws)

    # Transform to RR and ARD
    rr_draws = np.exp(theta)
    p_t_draws = p_c * rr_draws
    ard_draws = p_t_draws - p_c    # absolute risk difference in proportion

    # Summaries: RR
    rr_median = np.median(rr_draws)
    rr_mean   = np.mean(rr_draws)
    rr_L, rr_U = np.quantile(rr_draws, [0.025, 0.975])

    print("\nPosterior RR:")
    print(f"  Median RR = {rr_median:.3f}")
    print(f"  Mean RR   = {rr_mean:.3f}")
    print(f"  95% CrI  ≈ [{rr_L:.3f}, {rr_U:.3f}]")

    # Summaries: ARD (in %)
    ard_median = np.median(ard_draws) * 100
    ard_mean   = np.mean(ard_draws) * 100
    ard_L, ard_U = np.quantile(ard_draws, [0.025, 0.975])
    ard_L *= 100
    ard_U *= 100

    print("\nPosterior ARD (Complete – Culprit-only):")
    print(f"  Median ARD = {ard_median:.2f}%")
    print(f"  Mean ARD   = {ard_mean:.2f}%")
    print(f"  95% CrI   ≈ [{ard_L:.2f}%, {ard_U:.2f}%]")

    # Probabilities for ARD thresholds (in %)
    thresholds = [-10, -8, -6, -4, -2, 0, 2]
    print("\nPosterior probabilities P(ARD ≤ t%) (based on draws):")
    for t in thresholds:
        prob = np.mean(ard_draws <= t/100.0)
        print(f"  P(ARD ≤ {t:+d}%) ≈ {prob:.4f}")

    # Half-eye style plot on ARD (%) scale, using KDE
    ard_pct = ard_draws * 100
    kde = gaussian_kde(ard_pct)
    x_grid = np.linspace(x_min, x_max, 600)
    dens = kde(x_grid)
    dens /= dens.max()
    scale = 0.8

    fig, ax = plt.subplots(figsize=(8, 3))
    color = plt.get_cmap("tab10")(3)   # same greenish palette as before

    ax.fill_between(x_grid, 0, dens*scale, color=color, alpha=0.75)
    ax.axvline(0.0, linestyle="--", linewidth=1.1, color="black")   # no-effect
    ax.axvline(-5.0, linestyle=":", linewidth=1.0, color="black")   # MCID at -5%

    ax.set_xlim(x_min, x_max)
    ax.set_xticks(np.arange(x_min, x_max+1e-9, 5.0))
    ax.set_xticklabels([f"{int(v)}%" for v in np.arange(x_min, x_max+1e-9, 5.0)])
    ax.set_yticks([])
    ax.grid(False)

    ax.set_ylabel("Posterior density")
    ax.set_xlabel("Absolute Risk Difference in Primary Composite (Complete – Culprit-only)")
    ax.set_title(f"Primary composite — ARD posterior\nPrior: {prior_name}")

    plt.tight_layout()
    plt.show()


# --------------------------------------------
# RUN ANALYSES FOR EACH PRIOR
# --------------------------------------------
for name, mu0, sd0 in priors:
    analyze_composite_prior(name, mu0, sd0)
