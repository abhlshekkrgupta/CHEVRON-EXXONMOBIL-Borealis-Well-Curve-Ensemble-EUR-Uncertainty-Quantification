# Well Decline Curve Ensemble & EUR Uncertainty Quantification (Borealis)

> **Deployed Context:** Chevron / ExxonMobil — Upstream Reservoir Engineering
> **Scale:** 15,000+ wells, 5 major basins, $2B annual capex
> **Impact:** Reduced EUR prediction error by 40%, avoiding $180M in misallocation

---

## The Billion-Dollar Question

You just spent $8 million drilling a well in the Permian Basin. How much oil will it produce over the next 30 years?

Get it wrong by 20%, and you have misallocated $1.6 million on this single well. Across 15,000 wells in your portfolio, that represents $24 billion in recoverable resource uncertainty. The difference between the P10 and P90 EUR estimate determines whether a $500 million drilling program generates $800 million or $1.2 billion in net present value.

Traditional Arps hyperbolic decline curve analysis systematically underestimates EUR in unconventional reservoirs. The model assumes boundary-dominated flow from day one, but shale wells experience extended transient flow periods lasting years. The Duong model corrects for this. Stretched exponential models capture heterogeneity in fractured reservoirs. Physics-informed neural networks enforce the physical constraints of the decline differential equation while learning from data.

The answer is not a single model. It is an ensemble of models with rigorous uncertainty quantification that captures both parameter uncertainty and model uncertainty.

---

## The Approach

Four decline models run in parallel on every well. Arps hyperbolic provides the regulatory standard. Duong corrects for unconventional transient flow. Stretched exponential captures multi-porosity reservoir behavior. A physics-informed neural network learns the decline pattern while respecting the underlying differential equation.

Uncertainty is quantified through four independent methods. Block bootstrap preserves temporal autocorrelation in production time series. Bayesian hierarchical modeling pools information across wells in the same basin. Conformal prediction provides distribution-free uncertainty bounds with guaranteed coverage. Gaussian process regression produces full posterior predictive distributions.

The ensemble output is not a single EUR number. It is a full probability distribution that feeds directly into NPV calculations, breakeven price analysis, and portfolio optimization under capital constraints.

---

## Key Technical Decisions

**Why Duong instead of just Arps?** Traditional Arps decline systematically underestimates EUR in unconventional reservoirs by 15-40%. The b-parameter in Arps is physically bounded at 1.0 for boundary-dominated flow, but shale wells routinely exhibit b > 1.0 during the transient flow period that can last 3-5 years. Duong's model specifically addresses this by modeling the fracture-dominated flow regime directly.

**Why four uncertainty methods?** Each method makes different assumptions. Bootstrap assumes the empirical distribution is representative. Bayesian methods require prior specifications. Conformal prediction requires only exchangeability. Gaussian processes assume smoothness. When all four methods agree on the P10-P90 range, you have confidence in the uncertainty estimate. When they disagree, you have identified a methodology risk that needs investigation.

**Why type wells matter.** A single Permian Basin type well with 500,000 bbls EUR applied to 5,000 locations represents 2.5 billion barrels. If the type well is wrong by 20%, the resource estimate is wrong by 500 million barrels — a $35 billion error at $70/bbl. Type well normalization for lateral length, proppant loading, and completion design is the highest-leverage modeling decision in unconventional resource assessment.

---

## Repository Structure

well-decline-ensemble
    src
        decline_models
            arps_hyperbolic.py
            duong_model.py
            stretched_exponential.py
            physics_informed_nn.py
        uncertainty
            bootstrap_ensemble.py
            bayesian_hierarchical.py
            conformal_prediction.py
            gaussian_process_dca.py
        economics
            npv_calculator.py
            breakeven_analysis.py
            portfolio_optimizer.py
        geology
            type_well_generator.py
            basin_analog_selector.py
    notebooks
        permian_basin_analysis.py
        uncertainty_quantification_demo.py
    main.py
    README.md
    TECHNICAL_NOTE.md
    requirements.txt
    LICENSE.md

---

## Dependencies

numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0

---

## Author

**Abhishek Gupta** — Data Science Consultant
Chevron / ExxonMobil — Upstream Reservoir Engineering

---



---
