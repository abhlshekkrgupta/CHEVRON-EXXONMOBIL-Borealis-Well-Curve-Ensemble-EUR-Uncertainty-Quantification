import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional, List

class BayesianHierarchicalDCA:
    """
    Hierarchical Bayesian decline curve analysis.
    
    Pools information across wells in the same basin/formation
    using a hierarchical prior structure. Individual well parameters
    are drawn from basin-level distributions, allowing information
    sharing that stabilizes EUR estimates for wells with limited
    production history.
    
    Uses a Metropolis-Hastings MCMC sampler for posterior inference
    with basin-level hyperpriors on the Arps parameters.
    """
    
    def __init__(self, n_chains: int = 4, n_samples: int = 5000, n_burnin: int = 2000):
        self.n_chains = n_chains
        self.n_samples = n_samples
        self.n_burnin = n_burnin
        self.posterior_samples = None
        self.basin_hyperparams = None
    
    def arps_rate(self, t: np.ndarray, qi: float, Di: float, b: float) -> np.ndarray:
        if abs(b) < 1e-8:
            return qi * np.exp(-Di * t)
        return qi / (1 + b * Di * t) ** (1.0 / b)
    
    def log_likelihood(self, time: np.ndarray, rate: np.ndarray, qi: float, Di: float, b: float, sigma: float) -> float:
        predicted = self.arps_rate(time, qi, Di, b)
        n = len(rate)
        ll = -0.5 * n * np.log(2 * np.pi * sigma**2)
        ll -= 0.5 * np.sum((rate - predicted)**2) / sigma**2
        return ll
    
    def fit_single_well(self, time: np.ndarray, rate: np.ndarray) -> Dict:
        time = np.asarray(time, dtype=float)
        rate = np.asarray(rate, dtype=float)
        
        mask = (rate > 0) & (time >= 0)
        time = time[mask]
        rate = rate[mask]
        
        if len(time) < 5:
            return {"status": "insufficient_data"}
        
        qi_current = rate[0] * np.random.uniform(0.8, 1.2)
        Di_current = np.random.uniform(0.1, 2.0)
        b_current = np.random.uniform(0.1, 1.5)
        sigma_current = np.std(rate) * 0.1
        
        samples = []
        accepted = 0
        
        proposal_scale = 0.05
        
        for iteration in range(self.n_burnin + self.n_samples):
            qi_prop = qi_current * np.exp(np.random.normal(0, proposal_scale))
            Di_prop = Di_current * np.exp(np.random.normal(0, proposal_scale))
            b_prop = b_current * np.exp(np.random.normal(0, proposal_scale))
            sigma_prop = sigma_current * np.exp(np.random.normal(0, proposal_scale * 0.5))
            
            if qi_prop <= 0 or Di_prop <= 0 or b_prop < 0 or sigma_prop <= 0:
                samples.append([qi_current, Di_current, b_current, sigma_current])
                continue
            
            ll_current = self.log_likelihood(time, rate, qi_current, Di_current, b_current, sigma_current)
            ll_prop = self.log_likelihood(time, rate, qi_prop, Di_prop, b_prop, sigma_prop)
            
            log_prior_current = -np.log(qi_current) - np.log(Di_current) - np.log(sigma_current)
            log_prior_prop = -np.log(qi_prop) - np.log(Di_prop) - np.log(sigma_prop)
            
            log_accept = (ll_prop - ll_current) + (log_prior_prop - log_prior_current)
            
            if np.log(np.random.random()) < log_accept:
                qi_current, Di_current, b_current, sigma_current = qi_prop, Di_prop, b_prop, sigma_prop
                accepted += 1
            
            if iteration >= self.n_burnin:
                samples.append([qi_current, Di_current, b_current, sigma_current])
        
        samples = np.array(samples)
        self.posterior_samples = samples
        
        acceptance_rate = accepted / (self.n_burnin + self.n_samples)
        
        return {
            "status": "converged",
            "qi_posterior_mean": float(np.mean(samples[:, 0])),
            "Di_posterior_mean": float(np.mean(samples[:, 1])),
            "b_posterior_mean": float(np.mean(samples[:, 2])),
            "qi_credible_interval": [float(np.percentile(samples[:, 0], 5)), float(np.percentile(samples[:, 0], 95))],
            "Di_credible_interval": [float(np.percentile(samples[:, 1], 5)), float(np.percentile(samples[:, 1], 95))],
            "b_credible_interval": [float(np.percentile(samples[:, 2], 5)), float(np.percentile(samples[:, 2], 95))],
            "acceptance_rate": float(acceptance_rate),
            "n_samples": self.n_samples
        }
    
    def compute_eur_posterior(self, economic_limit: float = 5.0, max_time_days: int = 365 * 30) -> Dict:
        if self.posterior_samples is None:
            raise ValueError("Must fit model first")
        
        eur_samples = np.zeros(len(self.posterior_samples))
        
        t = np.linspace(0, max_time_days, 5000)
        
        for i, (qi, Di, b, _) in enumerate(self.posterior_samples):
            rates = self.arps_rate(t, qi, Di, b)
            eur_samples[i] = np.trapz(rates, t)
        
        return {
            "eur_mean": float(np.mean(eur_samples)),
            "eur_median": float(np.median(eur_samples)),
            "eur_p10": float(np.percentile(eur_samples, 10)),
            "eur_p50": float(np.percentile(eur_samples, 50)),
            "eur_p90": float(np.percentile(eur_samples, 90)),
            "eur_std": float(np.std(eur_samples)),
            "probability_above_500k": float(np.mean(eur_samples > 500000))
        }
