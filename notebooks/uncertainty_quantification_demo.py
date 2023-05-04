"""
Uncertainty Quantification Demo
================================
Comparison of bootstrap, Bayesian, conformal, and Gaussian Process
methods for EUR uncertainty quantification on a single well.
Demonstrates the impact of method choice on reserve booking.
"""

import numpy as np
from src.decline_models.arps_hyperbolic import ArpsHyperbolic
from src.uncertainty.bootstrap_ensemble import BootstrapEnsemble
from src.uncertainty.bayesian_hierarchical import BayesianHierarchicalDCA
from src.uncertainty.conformal_prediction import ConformalPrediction
from src.uncertainty.gaussian_process_dca import GaussianProcessDCA

np.random.seed(42)

print("=" * 60)
print("EUR UNCERTAINTY QUANTIFICATION METHODS COMPARISON")
print("=" * 60)

time = np.array([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 365,
                 395, 425, 455, 485, 515, 545, 575, 605, 635, 665, 695, 730])
rate = np.array([920, 780, 670, 580, 510, 452, 403, 362, 327, 297, 271, 248,
                 228, 210, 194, 180, 167, 155, 145, 135, 127, 119, 112, 105])

print(f"
  Well data: {len(time)} monthly observations")
print(f"  Initial rate: {rate[0]:.0f} bbl/day")
print(f"  Current rate: {rate[-1]:.0f} bbl/day")

methods = {}

print("
[1/4] Bootstrap ensemble...")
bootstrap = BootstrapEnsemble(n_bootstraps=3000, block_length=4)

def fit_wrapper(t, r):
    m = ArpsHyperbolic()
    m.fit(t, r)
    return m

def eur_wrapper(m):
    return m.compute_eur()

bootstrap_result = bootstrap.moving_block_bootstrap(time, rate, fit_wrapper, eur_wrapper)
methods["Bootstrap"] = bootstrap_result
print(f"  EUR: {bootstrap_result['mean_eur']:,.0f} (P10-P90: {bootstrap_result['p10_eur']:,.0f} - {bootstrap_result['p90_eur']:,.0f})")

print("
[2/4] Bayesian hierarchical...")
bayesian = BayesianHierarchicalDCA(n_samples=3000, n_burnin=1000)
bayesian_fit = bayesian.fit_single_well(time, rate)
bayesian_eur = bayesian.compute_eur_posterior()
methods["Bayesian"] = bayesian_eur
print(f"  EUR: {bayesian_eur['eur_mean']:,.0f} (P10-P90: {bayesian_eur['eur_p10']:,.0f} - {bayesian_eur['eur_p90']:,.0f})")

print("
[3/4] Conformal prediction...")
conformal = ConformalPrediction(confidence_level=0.90)
train_size = 18
conformal_result = conformal.adaptive_conformal(
    time, rate, fit_wrapper,
    lambda m, t: m.hyperbolic_rate(t, m.qi, m.Di, m.b),
    window_size=6
)
arps = ArpsHyperbolic()
arps.fit(time, rate)
arps_eur = arps.compute_eur()
conformal_eur = conformal.compute_eur_conformal(arps_eur["eur_barrels"], bootstrap_result.get("eur_samples", bootstrap_result.get("mean_eur", np.array([arps_eur["eur_barrels"]]))))
methods["Conformal"] = conformal_eur
print(f"  EUR: {conformal_eur['eur_point_estimate']:,.0f} (bounds: {conformal_eur['eur_lower_bound']:,.0f} - {conformal_eur['eur_upper_bound']:,.0f})")

print("
[4/4] Gaussian Process...")
gp = GaussianProcessDCA(kernel_length_scale=100, kernel_variance=50000, noise_variance=500)
gp_fit = gp.fit(time.reshape(-1, 1), rate)
gp_eur = gp.compute_eur_posterior(n_samples=3000)
methods["Gaussian Process"] = gp_eur
print(f"  EUR: {gp_eur['eur_mean']:,.0f} (P10-P90: {gp_eur['eur_p10']:,.0f} - {gp_eur['eur_p90']:,.0f})")

print(f"
{'='*60}")
print("METHODS COMPARISON SUMMARY")
print(f"{'='*60}")

print(f"
{'Method':<20} {'Mean EUR':>12} {'P10 EUR':>12} {'P90 EUR':>12} {'P90-P10':>12}")
print("-" * 70)

for method_name, result in methods.items():
    if "mean_eur" in result or "eur_mean" in result:
        mean = result.get("mean_eur") or result.get("eur_mean") or result.get("eur_point_estimate", 0)
        p10 = result.get("p10_eur") or result.get("eur_p10") or result.get("eur_lower_bound", 0)
        p90 = result.get("p90_eur") or result.get("eur_p90") or result.get("eur_upper_bound", 0)
        spread = p90 - p10
        print(f"{method_name:<20} {mean:>12,.0f} {p10:>12,.0f} {p90:>12,.0f} {spread:>12,.0f}")

all_means = [r.get("mean_eur") or r.get("eur_mean") or r.get("eur_point_estimate", 0) for r in methods.values()]
ensemble_mean = np.mean(all_means)
ensemble_std = np.std(all_means)

print(f"
  Ensemble mean EUR: {ensemble_mean:,.0f} bbls")
print(f"  Between-method std: {ensemble_std:,.0f} bbls ({ensemble_std/ensemble_mean*100:.1f}%)")

print(f"
{'='*60}")
print("RECOMMENDATION: Report ensemble of all four methods.")
print("Bootstrap for regulatory reserve booking (SEC standard).")
print("Bayesian for internal resource assessment.")
print("Conformal for distribution-free uncertainty bounds.")
print("Gaussian Process when smooth decline assumed.")
print(f"{'='*60}")
