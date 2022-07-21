import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
from typing import Dict, Tuple, Optional, List

class ArpsHyperbolic:
    """
    Arps hyperbolic decline curve analysis with rigorous uncertainty.
    
    The industry standard for conventional reservoir decline analysis.
    Models production rate q(t) = qi / (1 + b * Di * t)^(1/b)
    where qi is initial rate, Di is initial decline rate, and b
    is the hyperbolic exponent (0 < b < 1 for most wells).
    
    For b=0, reduces to exponential decline. For b=1, harmonic.
    Unconventional wells often exhibit b > 1 in early time,
    which this model handles through segmented fitting.
    """
    
    def __init__(self):
        self.qi = None
        self.Di = None
        self.b = None
        self.parameter_covariance = None
        self.eur = None
        self.eur_uncertainty = None
    
    def hyperbolic_rate(self, t: np.ndarray, qi: float, Di: float, b: float) -> np.ndarray:
        if abs(b) < 1e-8:
            return qi * np.exp(-Di * t)
        
        return qi / (1 + b * Di * t) ** (1.0 / b)
    
    def fit(self, time: np.ndarray, rate: np.ndarray) -> Dict:
        time = np.asarray(time, dtype=float)
        rate = np.asarray(rate, dtype=float)
        
        mask = (rate > 0) & (time >= 0)
        time = time[mask]
        rate = rate[mask]
        
        if len(time) < 5:
            return {"status": "insufficient_data"}
        
        qi_guess = rate[0]
        Di_guess = 0.5
        
        if len(time) > 1 and time[-1] > 0:
            Di_guess = (rate[0] - rate[-1]) / (rate[0] * time[-1])
            Di_guess = max(0.001, min(Di_guess, 2.0))
        
        b_guess = 0.5
        
        try:
            popt, pcov = curve_fit(
                self.hyperbolic_rate, time, rate,
                p0=[qi_guess, Di_guess, b_guess],
                bounds=([0, 0.0001, 0], [np.inf, 5.0, 2.0]),
                maxfev=5000
            )
            
            self.qi, self.Di, self.b = popt
            self.parameter_covariance = pcov
            
            param_errors = np.sqrt(np.diag(pcov))
            
            fitted_rates = self.hyperbolic_rate(time, self.qi, self.Di, self.b)
            residuals = rate - fitted_rates
            rmse = np.sqrt(np.mean(residuals ** 2))
            
            n = len(time)
            p = 3
            aic = n * np.log(np.sum(residuals ** 2) / n) + 2 * p
            
            return {
                "status": "converged",
                "qi": float(self.qi),
                "Di": float(self.Di),
                "b": float(self.b),
                "qi_se": float(param_errors[0]),
                "Di_se": float(param_errors[1]),
                "b_se": float(param_errors[2]),
                "rmse": float(rmse),
                "aic": float(aic),
                "n_points": n
            }
        
        except Exception as e:
            return {"status": "fit_failed", "error": str(e)}
    
    def compute_eur(self, economic_limit: float = 5.0, max_time_days: int = 365 * 30) -> Dict:
        if self.qi is None:
            raise ValueError("Must fit model before computing EUR")
        
        t = np.linspace(0, max_time_days, 10000)
        rates = self.hyperbolic_rate(t, self.qi, self.Di, self.b)
        
        mask = rates >= economic_limit
        if not np.any(mask):
            t_economic = t[-1]
        else:
            t_economic = t[mask][-1]
        
        dt = t[1] - t[0]
        cumulative = np.trapz(rates, t)
        
        self.eur = cumulative
        self.economic_life = t_economic
        
        return {
            "eur_barrels": float(cumulative),
            "economic_life_years": float(t_economic / 365),
            "economic_limit_bbl_day": economic_limit,
            "final_rate_bbl_day": float(rates[-1])
        }
    
    def estimate_eur_uncertainty(self, n_simulations: int = 5000) -> Dict:
        if self.parameter_covariance is None:
            raise ValueError("Must fit model before uncertainty estimation")
        
        param_means = np.array([self.qi, self.Di, self.b])
        param_cov = self.parameter_covariance
        
        if np.any(np.diag(param_cov) < 0):
            param_cov = np.abs(param_cov)
        
        eur_samples = np.zeros(n_simulations)
        
        for i in range(n_simulations):
            params = np.random.multivariate_normal(param_means, param_cov)
            qi_s, Di_s, b_s = params
            
            if qi_s <= 0 or Di_s <= 0 or b_s < 0:
                continue
            
            t = np.linspace(0, 365 * 30, 5000)
            rates = self.hyperbolic_rate(t, qi_s, Di_s, b_s)
            eur_samples[i] = np.trapz(rates, t)
        
        eur_samples = eur_samples[eur_samples > 0]
        
        self.eur_uncertainty = {
            "mean_eur": float(np.mean(eur_samples)),
            "median_eur": float(np.median(eur_samples)),
            "p10_eur": float(np.percentile(eur_samples, 10)),
            "p50_eur": float(np.percentile(eur_samples, 50)),
            "p90_eur": float(np.percentile(eur_samples, 90)),
            "std_eur": float(np.std(eur_samples)),
            "coefficient_of_variation": float(np.std(eur_samples) / np.mean(eur_samples)) if np.mean(eur_samples) > 0 else 0
        }
        
        return self.eur_uncertainty
