"""
Permian Basin Analysis
=======================
Comprehensive EUR estimation for Permian Basin wells using
ensemble decline curve analysis with rigorous uncertainty
quantification across the Midland and Delaware sub-basins.
"""

import numpy as np
from src.decline_models.arps_hyperbolic import ArpsHyperbolic
from src.decline_models.duong_model import DuongModel
from src.decline_models.stretched_exponential import StretchedExponential
from src.uncertainty.bootstrap_ensemble import BootstrapEnsemble
from src.uncertainty.conformal_prediction import ConformalPrediction
from src.economics.npv_calculator import NPVCalculator
from src.economics.breakeven_analysis import BreakevenAnalysis

np.random.seed(42)

print("=" * 60)
print("PERMIAN BASIN EUR ANALYSIS")
print("=" * 60)

print("
[1/4] Fitting decline models to Permian well data...")

time_midland = np.array([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 365,
                         395, 425, 455, 485, 515, 545, 575, 605, 635, 665, 695, 730])
rate_midland = np.array([850, 720, 620, 540, 480, 430, 390, 355, 325, 300, 278, 258,
                         240, 224, 210, 197, 185, 174, 164, 155, 146, 138, 131, 124])

time_delaware = np.array([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 365,
                          395, 425, 455, 485, 515, 545, 575, 605, 635, 665, 695, 730])
rate_delaware = np.array([1100, 900, 760, 650, 560, 490, 430, 382, 340, 305, 275, 249,
                          226, 206, 188, 172, 158, 145, 134, 124, 115, 107, 100, 93])

arps_midland = ArpsHyperbolic()
arps_fit_midland = arps_midland.fit(time_midland, rate_midland)
eur_arps_midland = arps_midland.compute_eur()

duong_midland = DuongModel()
duong_fit_midland = duong_midland.fit(time_midland, rate_midland)
eur_duong_midland = duong_midland.compute_eur()

print(f"  Midland Basin:")
print(f"    Arps EUR: {eur_arps_midland['eur_barrels']:,.0f} bbls")
print(f"    Duong EUR: {eur_duong_midland['eur_barrels']:,.0f} bbls")

arps_delaware = ArpsHyperbolic()
arps_fit_delaware = arps_delaware.fit(time_delaware, rate_delaware)
eur_arps_delaware = arps_delaware.compute_eur()

duong_delaware = DuongModel()
duong_fit_delaware = duong_delaware.fit(time_delaware, rate_delaware)
eur_duong_delaware = duong_delaware.compute_eur()

print(f"  Delaware Basin:")
print(f"    Arps EUR: {eur_arps_delaware['eur_barrels']:,.0f} bbls")
print(f"    Duong EUR: {eur_duong_delaware['eur_barrels']:,.0f} bbls")

print("
[2/4] Running bootstrap uncertainty quantification...")

bootstrap = BootstrapEnsemble(n_bootstraps=3000, block_length=4)

def fit_arps_wrapper(time, rate):
    model = ArpsHyperbolic()
    model.fit(time, rate)
    return model

def eur_wrapper(model):
    return model.compute_eur()

bootstrap_midland = bootstrap.moving_block_bootstrap(
    time_midland, rate_midland, fit_arps_wrapper, eur_wrapper
)

print(f"  Midland Bootstrap Results:")
print(f"    Mean EUR: {bootstrap_midland['mean_eur']:,.0f} bbls")
print(f"    P10-P90 range: {bootstrap_midland['p10_eur']:,.0f} - {bootstrap_midland['p90_eur']:,.0f}")
print(f"    CV: {bootstrap_midland['cv']:.2%}")

print("
[3/4] Computing conformal prediction intervals...")

conformal = ConformalPrediction(confidence_level=0.90)

train_size = 18
conformal_result = conformal.adaptive_conformal(
    time_midland, rate_midland,
    fit_arps_wrapper,
    lambda model, t: model.hyperbolic_rate(t, model.qi, model.Di, model.b),
    window_size=6
)

print(f"  Conformal Prediction:")
print(f"    q_hat: {conformal_result.get('q_hat', 0):.1f} bbl/day")
print(f"    Empirical coverage: {conformal_result.get('empirical_coverage', 0):.1%}")

print("
[4/4] Computing well economics...")

npv_calc = NPVCalculator(discount_rate=0.10)
npv_calc.set_price_forecast(75.0, 3.50, 25.0)
npv_calc.set_cost_structure(5000, 3.50, 0.20, 0.05, 0.02)

annual_prod = np.exp(-0.3 * np.arange(30))
annual_prod = annual_prod / np.sum(annual_prod) * eur_duong_midland["eur_barrels"]

npv_midland = npv_calc.compute_npv(annual_prod, 8500000)

breakeven = BreakevenAnalysis()
be_midland = breakeven.compute_breakeven_price(
    eur_duong_midland["eur_barrels"], 8500000
)

print(f"  Midland Well Economics:")
print(f"    NPV at $75/bbl: ${npv_midland['npv']:,.0f}")
print(f"    Breakeven price: ${be_midland['breakeven_price_per_bbl']:.2f}/bbl")
print(f"    Payback year: {npv_midland['payback_year']}")

print(f"
{'='*60}")
print("PERMIAN ANALYSIS COMPLETE")
print(f"{'='*60}")
