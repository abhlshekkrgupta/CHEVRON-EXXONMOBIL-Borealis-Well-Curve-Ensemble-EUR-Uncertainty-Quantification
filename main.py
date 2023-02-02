"""
Borealis — Well Decline Curve Ensemble & EUR Uncertainty Quantification
Chevron / ExxonMobil — Upstream Reservoir Engineering

Execution entry point for multi-model decline curve analysis,
uncertainty quantification, and well economics across 15,000+
wells in 5 major basins.
"""

import numpy as np
from src.decline_models.arps_hyperbolic import ArpsHyperbolic
from src.decline_models.duong_model import DuongModel
from src.decline_models.stretched_exponential import StretchedExponential
from src.decline_models.physics_informed_nn import PhysicsInformedNN
from src.uncertainty.bootstrap_ensemble import BootstrapEnsemble
from src.uncertainty.bayesian_hierarchical import BayesianHierarchicalDCA
from src.uncertainty.conformal_prediction import ConformalPrediction
from src.uncertainty.gaussian_process_dca import GaussianProcessDCA
from src.economics.npv_calculator import NPVCalculator
from src.economics.breakeven_analysis import BreakevenAnalysis
from src.economics.portfolio_optimizer import PortfolioOptimizer
from src.geology.type_well_generator import TypeWellGenerator
from src.geology.basin_analog_selector import BasinAnalogSelector

np.random.seed(42)

print("=" * 65)
print("  BOREALIS — Well Decline Curve Ensemble & EUR Uncertainty")
print("  Chevron / ExxonMobil — Upstream Reservoir Engineering")
print("=" * 65)

print("
[1/6] Fitting multi-model decline ensemble...")

time_permian = np.array([30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 365,
                         395, 425, 455, 485, 515, 545, 575, 605, 635, 665, 695, 730])
rate_permian = np.array([950, 800, 680, 590, 510, 450, 395, 350, 315, 285, 260, 238,
                         218, 200, 185, 171, 159, 148, 138, 129, 121, 114, 107, 101])

models = {}

arps = ArpsHyperbolic()
arps_result = arps.fit(time_permian, rate_permian)
arps_eur = arps.compute_eur()
models["Arps"] = arps_eur["eur_barrels"]
print(f"  Arps Hyperbolic: EUR = {arps_eur['eur_barrels']:,.0f} bbls (b={arps.b:.3f})")

duong = DuongModel()
duong_result = duong.fit(time_permian, rate_permian)
duong_eur = duong.compute_eur()
models["Duong"] = duong_eur["eur_barrels"]
print(f"  Duong: EUR = {duong_eur['eur_barrels']:,.0f} bbls (m={duong.m:.3f})")

stretched = StretchedExponential()
stretched_result = stretched.fit(time_permian, rate_permian)
stretched_eur = stretched.compute_eur()
models["StretchedExp"] = stretched_eur["eur_barrels"]
print(f"  Stretched Exp: EUR = {stretched_eur['eur_barrels']:,.0f} bbls (beta={stretched.beta:.3f})")

pinn = PhysicsInformedNN(hidden_units=32, n_layers=3)
pinn_result = pinn.fit(time_permian, rate_permian, n_iterations=3000)
pinn_eur = pinn.compute_eur(time_norm=730, rate_norm=1000)
models["PINN"] = pinn_eur["eur_barrels"]
print(f"  Physics-Informed NN: EUR = {pinn_eur['eur_barrels']:,.0f} bbls")

ensemble_eur = np.mean(list(models.values()))
ensemble_std = np.std(list(models.values()))
print(f"
  Ensemble EUR: {ensemble_eur:,.0f} +/- {ensemble_std:,.0f} bbls ({ensemble_std/ensemble_eur*100:.1f}%)")

print("
[2/6] Running uncertainty quantification...")

bootstrap = BootstrapEnsemble(n_bootstraps=3000, block_length=4)

def fit_arps_wrapper(t, r):
    m = ArpsHyperbolic()
    m.fit(t, r)
    return m

def eur_wrapper(m):
    return m.compute_eur()

bootstrap_result = bootstrap.moving_block_bootstrap(
    time_permian, rate_permian, fit_arps_wrapper, eur_wrapper
)
print(f"  Bootstrap: P10={bootstrap_result['p10_eur']:,.0f}, P50={bootstrap_result['p50_eur']:,.0f}, P90={bootstrap_result['p90_eur']:,.0f}")

bayesian = BayesianHierarchicalDCA(n_samples=3000, n_burnin=1000)
bayesian_fit = bayesian.fit_single_well(time_permian, rate_permian)
bayesian_eur = bayesian.compute_eur_posterior()
print(f"  Bayesian: P10={bayesian_eur['eur_p10']:,.0f}, P50={bayesian_eur['eur_p50']:,.0f}, P90={bayesian_eur['eur_p90']:,.0f}")

gp = GaussianProcessDCA()
gp.fit(time_permian.reshape(-1, 1), rate_permian)
gp_eur = gp.compute_eur_posterior(n_samples=2000)
print(f"  Gaussian Process: P10={gp_eur['eur_p10']:,.0f}, P50={gp_eur['eur_p50']:,.0f}, P90={gp_eur['eur_p90']:,.0f}")

print("
[3/6] Computing well economics...")

npv_calc = NPVCalculator(discount_rate=0.10)
npv_calc.set_price_forecast(72.50, 3.50, 25.0)
npv_calc.set_cost_structure(5000, 3.50, 0.20, 0.05, 0.02)

annual_prod = np.exp(-0.3 * np.arange(30))
annual_prod = annual_prod / np.sum(annual_prod) * ensemble_eur

npv_result = npv_calc.compute_npv(annual_prod, 8000000)
print(f"  NPV at $72.50/bbl: ${npv_result['npv']:,.0f}")
print(f"  Profitability index: {npv_result['profitability_index']:.2f}")
print(f"  Payback year: {npv_result['payback_year']}")

breakeven = BreakevenAnalysis()
be_result = breakeven.compute_breakeven_price(ensemble_eur, 8000000)
print(f"  Breakeven oil price: ${be_result['breakeven_price_per_bbl']:.2f}/bbl")

print("
[4/6] Running portfolio optimization...")

portfolio = PortfolioOptimizer(annual_budget=500000000)

basins = ["Permian_Midland", "Permian_Delaware", "Eagle_Ford", "Bakken", "DJ_Basin"]
for i in range(20):
    basin = basins[i % len(basins)]
    eur_mean = np.random.uniform(300000, 1200000)
    
    portfolio.add_well_candidate(
        well_id=f"WELL_{i:04d}",
        basin=basin,
        drill_cost=np.random.uniform(6000000, 12000000),
        expected_eur=eur_mean,
        eur_std=eur_mean * 0.30,
        expected_npv=np.random.uniform(1000000, 15000000),
        npv_std=eur_mean * 0.30 * 30,
        success_probability=0.95,
        correlation_group=basin
    )

selection = portfolio.greedy_portfolio_selection()
simulation = portfolio.simulate_portfolio_outcomes(n_simulations=5000)

print(f"  Wells selected: {selection['n_wells_selected']}")
print(f"  Capital deployed: ${selection['total_capital_deployed']:,.0f}")
print(f"  Expected portfolio NPV: ${simulation['expected_npv']:,.0f}")
print(f"  NPV P10-P90: ${simulation['npv_p10']:,.0f} - ${simulation['npv_p90']:,.0f}")

print("
[5/6] Generating type well for Permian Basin...")

type_well_gen = TypeWellGenerator("Permian", "Wolfcamp_A")

for i in range(15):
    lateral = np.random.uniform(7000, 12000)
    eur = 400000 + lateral * 45 + np.random.normal(0, 50000)
    peak = 800 + lateral * 0.07 + np.random.normal(0, 50)
    profile = peak * np.exp(-0.3 * np.arange(36))
    
    type_well_gen.add_offset_well(
        well_id=f"OFFSET_{i:03d}",
        lateral_length_ft=lateral,
        proppant_lbs_per_ft=np.random.uniform(1500, 2500),
        fluid_bbls_per_ft=np.random.uniform(40, 60),
        stages=int(lateral / 200),
        peak_rate_bbl_day=peak,
        eur_barrels=eur,
        production_profile=profile
    )

type_well = type_well_gen.normalize_to_type_well(10000, 2000, 50)
print(f"  Type well EUR: P50={type_well['type_well']['p50_eur']:,.0f} bbls")
print(f"  P10-P90 range: {type_well['type_well']['p10_eur']:,.0f} - {type_well['type_well']['p90_eur']:,.0f}")

resource = type_well_gen.compute_resource_potential(5000)
print(f"  Total resource: {resource['total_resource_mean_mmbo']:.1f} MMBO")

print("
[6/6] Finding basin analogs...")

analog_selector = BasinAnalogSelector()

analog_selector.register_mature_basin(
    "Permian_Midland", "Wolfcamp_A",
    {"depth_ft": 8500, "thickness_ft": 350, "porosity_pct": 8.5, "permeability_md": 0.005, "pressure_psi": 5500, "thermal_maturity_ro": 1.2, "clay_content_pct": 12, "organic_richness_toc": 4.5},
    {"mean_eur": 650000, "p10_eur": 420000, "p90_eur": 920000}, 2500
)

analog_selector.register_mature_basin(
    "Permian_Delaware", "Wolfcamp_A",
    {"depth_ft": 9500, "thickness_ft": 280, "porosity_pct": 7.5, "permeability_md": 0.003, "pressure_psi": 6200, "thermal_maturity_ro": 1.4, "clay_content_pct": 15, "organic_richness_toc": 5.2},
    {"mean_eur": 720000, "p10_eur": 480000, "p90_eur": 1050000}, 1800
)

analog_selector.register_mature_basin(
    "Eagle_Ford", "Lower_Eagle_Ford",
    {"depth_ft": 11000, "thickness_ft": 200, "porosity_pct": 10.0, "permeability_md": 0.008, "pressure_psi": 7500, "thermal_maturity_ro": 1.5, "clay_content_pct": 8, "organic_richness_toc": 3.8},
    {"mean_eur": 550000, "p10_eur": 350000, "p90_eur": 800000}, 3200
)

target_geology = {"depth_ft": 9200, "thickness_ft": 300, "porosity_pct": 8.0, "permeability_md": 0.004, "pressure_psi": 5800, "thermal_maturity_ro": 1.3, "clay_content_pct": 13, "organic_richness_toc": 4.8}

analogs = analog_selector.find_analogs(target_geology, top_n=3)
print(f"  Best analog: {analogs['analogs'][0]['basin']} ({analogs['analogs'][0]['formation']})")
print(f"  Similarity score: {analogs['analogs'][0]['similarity_score']:.3f}")
print(f"  Weighted EUR estimate: {analogs['weighted_eur_mean']:,.0f} bbls")

print("
" + "=" * 65)
print("  BOREALIS ANALYSIS COMPLETE")
print("=" * 65)

print(f"
  Key outcomes:")
print(f"    1. Ensemble EUR: {ensemble_eur:,.0f} bbls (across 4 models)")
print(f"    2. Uncertainty range: ${bootstrap_result['p90_eur'] - bootstrap_result['p10_eur']:,.0f} bbls (P10-P90)")
print(f"    3. Well NPV: ${npv_result['npv']:,.0f} at $72.50/bbl")
print(f"    4. Portfolio NPV: ${simulation['expected_npv']:,.0f} ({selection['n_wells_selected']} wells)")
print(f"    5. Basin resource: {resource['total_resource_mean_mmbo']:.1f} MMBO")
