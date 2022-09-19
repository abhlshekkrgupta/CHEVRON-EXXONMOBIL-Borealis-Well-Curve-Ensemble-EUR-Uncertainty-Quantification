import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, Tuple, Optional

class StretchedExponential:
    """
    Stretched exponential (power-law) decline model.
    
    q(t) = qi * exp(-(t / tau)^beta)
    
    Where tau is the characteristic time scale and beta is the
    stretching exponent. When beta=1, reduces to standard exponential.
    When beta<1, produces the stretched exponential behavior observed
    in heterogeneous reservoirs with multi-porosity systems.
    
    Particularly useful for naturally fractured reservoirs where
    the fracture network creates multiple characteristic time scales
    that cannot be captured by a single exponential decay.
    """
    
    def __init__(self):
        self.qi = None
        self.tau = None
        self.beta = None
        self.fitted = False
    
    def stretched_exponential_rate(self, t: np.ndarray, qi: float, tau: float, beta: float) -> np.ndarray:
        return qi * np.exp(-(t / tau) ** beta)
    
    def fit(self, time: np.ndarray, rate: np.ndarray) -> Dict:
        time = np.asarray(time, dtype=float)
        rate = np.asarray(rate, dtype=float)
        
        mask = (rate > 0) & (time >= 0)
        time = time[mask]
        rate = rate[mask]
        
        if len(time) < 5:
            return {"status": "insufficient_data"}
        
        qi_guess = rate[0]
        
        half_life_idx = np.argmin(np.abs(rate - qi_guess / 2))
        tau_guess = time[half_life_idx] if half_life_idx < len(time) else time[len(time)//2]
        tau_guess = max(1.0, tau_guess)
        
        beta_guess = 0.7
        
        try:
            popt, pcov = curve_fit(
                self.stretched_exponential_rate, time, rate,
                p0=[qi_guess, tau_guess, beta_guess],
                bounds=([0, 1.0, 0.1], [np.inf, 365*50, 1.5]),
                maxfev=5000
            )
            
            self.qi, self.tau, self.beta = popt
            self.fitted = True
            
            param_errors = np.sqrt(np.diag(pcov))
            
            fitted_rates = self.stretched_exponential_rate(time, self.qi, self.tau, self.beta)
            residuals = rate - fitted_rates
            rmse = np.sqrt(np.mean(residuals ** 2))
            
            return {
                "status": "converged",
                "qi": float(self.qi),
                "tau_days": float(self.tau),
                "beta": float(self.beta),
                "qi_se": float(param_errors[0]),
                "tau_se": float(param_errors[1]),
                "beta_se": float(param_errors[2]),
                "rmse": float(rmse),
                "n_points": len(time),
                "characteristic_time_years": float(self.tau / 365)
            }
        
        except Exception as e:
            return {"status": "fit_failed", "error": str(e)}
    
    def compute_eur(self, economic_limit: float = 5.0, max_time_days: int = 365 * 30) -> Dict:
        if not self.fitted:
            raise ValueError("Must fit model before computing EUR")
        
        t = np.linspace(0, max_time_days, 10000)
        rates = self.stretched_exponential_rate(t, self.qi, self.tau, self.beta)
        
        cumulative = np.trapz(rates, t)
        
        economic_mask = rates >= economic_limit
        t_economic = t[economic_mask][-1] if np.any(economic_mask) else t[-1]
        
        half_life = self.tau * (np.log(2)) ** (1.0 / self.beta)
        
        return {
            "eur_barrels": float(cumulative),
            "economic_life_years": float(t_economic / 365),
            "half_life_days": float(half_life),
            "economic_limit_bbl_day": economic_limit
        }
