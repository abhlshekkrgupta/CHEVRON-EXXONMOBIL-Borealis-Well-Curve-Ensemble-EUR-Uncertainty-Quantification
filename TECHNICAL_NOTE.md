# Technical Note: Decline Curve Ensemble & EUR Uncertainty

**Author:** Abhishek Gupta
**Context:** Chevron / ExxonMobil — Upstream Reservoir Engineering

---

## Decline Curve Model Selection

### Arps Hyperbolic (1945)

The industry standard for conventional reservoirs. The rate-time relationship is q(t) = qi / (1 + b * Di * t)^(1/b). For b=0, reduces to exponential decline. For b=1, harmonic decline. Valid for boundary-dominated flow. Underestimates EUR in unconventional wells by 15-40% because it cannot capture extended transient flow.

### Duong Model (2011)

Developed specifically for fracture-dominated unconventional reservoirs. The rate-time relationship accounts for the linear flow regime that dominates early production in hydraulically fractured horizontal wells. The m parameter characterizes the fracture network conductivity. This is the model actually used in Permian Basin well evaluation.

### Stretched Exponential

q(t) = qi * exp(-(t/tau)^beta). When beta<1, produces the stretched exponential behavior observed in heterogeneous reservoirs with multi-porosity systems. The characteristic time tau and stretching exponent beta capture both the early-time fracture flow and late-time matrix contribution.

### Physics-Informed Neural Network

A feedforward neural network trained with a composite loss function that includes both data fidelity and physics constraint terms. The physics loss penalizes violations of the hyperbolic decline differential equation dq/dt = -D * q^b. Produces physically consistent forecasts even with sparse or noisy data.

---

## Uncertainty Quantification Methods

### Block Bootstrap

Standard bootstrap assumes independent observations. Production time series exhibit strong autocorrelation. Moving block bootstrap preserves temporal dependence by resampling contiguous blocks. Optimal block length is estimated using the Politis-White algorithm. The stationary bootstrap variant uses random block lengths drawn from a geometric distribution.

### Bayesian Hierarchical DCA

Pools information across wells through hierarchical priors. Individual well parameters are drawn from basin-level distributions with hyperpriors. Metropolis-Hastings MCMC samples the posterior. Information sharing stabilizes EUR estimates for wells with limited production history.

### Conformal Prediction

Distribution-free uncertainty quantification with guaranteed coverage under exchangeability. Split conformal prediction uses a calibration set to determine prediction interval width. The q-hat value is the (n+1)*(confidence_level)-th order statistic of calibration residuals. Valid without Gaussian or any other distributional assumptions.

### Gaussian Process DCA

Treats the decline curve as a latent function with a Matern kernel prior. The Matern kernel with nu=3/2 produces once-differentiable functions suitable for smooth decline. Posterior predictive distributions capture both aleatoric noise and epistemic uncertainty from limited observations.

---

## Reserve Booking Implications

SEC reserves definitions require "reasonable certainty" for proved reserves, typically interpreted as P90 confidence. Different uncertainty methods produce different P90 estimates. The ensemble approach reports the range across methods. When methods disagree materially, the most conservative estimate is used for SEC reporting while the ensemble mean informs internal resource assessment.

---

## Author

**Abhishek Gupta**
Data Science Consultant
Chevron / ExxonMobil — Upstream Reservoir Engineering

**Proprietary and Confidential**
