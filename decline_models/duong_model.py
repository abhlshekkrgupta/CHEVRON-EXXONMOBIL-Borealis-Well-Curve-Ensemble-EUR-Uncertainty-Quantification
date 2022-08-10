import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
from typing import Dict, Tuple, Optional

class DuongModel:
    """
    Duong decline model for unconventional tight oil and shale gas reservoirs.
    
    Traditional Arps hyperbolic decline systematically underestimates EUR
    in unconventional reservoirs because it assumes boundary-dominated flow
    from day one. Unconventional wells experience extended transient flow
    periods that Arps cannot capture — the b-parameter often exceeds 1.0
    in early time, which is physically impossible under Arps assumptions.
    
    Duong (2011) proposed a rate-time relationship specifically for
    fracture-dominated flow: q(t) = qi * t^(-m) * exp((a/(1-m)) * (t^(1-m) - 1))
    where m is the decline exponent and a is the intercept coefficient.
    
    This is the model actually used in Permian Basin well analysis.
    """
    
    def __init__(self):
        self.qi = None
        self.a = None
        self.m = None
        self.fitted = False
    
    def duong_rate(self, t: np.ndarray, qi: float, a: float, m: float) -> np.ndarray:
        t = np.maximum(t, 0.001)
        
        if abs(m - 1.0) < 1e-6:
            return qi * np.exp(a * np.log(t))
        
        return qi * t ** (-m) * np.exp((a / (1 - m)) * (t ** (1 - m) - 1))
    
    def fit(self, time: np.ndarray, rate: np.ndarray) -> Dict:
        time = np.asarray(time, dtype=float)
        rate = np.asarray(rate, dtype=float)
        
        mask = (rate > 0) & (time > 0)
        time = time[mask]
        rate = rate[mask]
        
        if len(time) < 5:
            return {"status": "insufficient_data"}
        
        log_time = np.log(time)
        log_rate = np.log(rate)
        
        slope, intercept = np.polyfit(log_time, log_rate, 1)
        m_guess = max(0.5, -slope)
        a_guess = abs(intercept) * 0.1
        qi_guess = rate[0]
        
        try:
            popt, pcov = curve_fit(
                self.duong_rate, time, rate,
                p0=[qi_guess, a_guess, m_guess],
                bounds=([0, 0.0001, 0.1], [np.inf, 5.0, 2.0]),
                maxfev=5000
            )
            
            self.qi, self.a, self.m = popt
            self.fitted = True
            
            param_errors = np.sqrt(np.diag(pcov))
            
            fitted_rates = self.duong_rate(time, self.qi, self.a, self.m)
            residuals = rate - fitted_rates
            rmse = np.sqrt(np.mean(residuals ** 2))
            
            return {
                "status": "converged",
                "qi": float(self.qi),
                "a": float(self.a),
                "m": float(self.m),
                "qi_se": float(param_errors[0]),
                "a_se": float(param_errors[1]),
                "m_se": float(param_errors[2]),
                "rmse": float(rmse),
                "n_points": len(time)
            }
        
        except Exception as e:
            return {"status": "fit_failed", "error": str(e)}
    
    def compute_eur(self, economic_limit: float = 5.0, max_time_days: int = 365 * 30) -> Dict:
        if not self.fitted:
            raise ValueError("Must fit model before computing EUR")
        
        t = np.linspace(0.01, max_time_days, 10000)
        rates = self.duong_rate(t, self.qi, self.a, self.m)
        
        dt = t[1] - t[0]
        cumulative = np.trapz(rates, t) * (t[1] - t[0]) if len(t) > 1 else 0
        
        cumulative = np.trapz(rates, t)
        
        economic_mask = rates >= economic_limit
        t_economic = t[economic_mask][-1] if np.any(economic_mask) else t[-1]
        
        return {
            "eur_barrels": float(cumulative),
            "economic_life_years": float(t_economic / 365),
            "economic_limit_bbl_day": economic_limit
        }
    
    def compare_to_arps(self, time: np.ndarray, rate: np.ndarray, arps_result: Dict) -> Dict:
        if not self.fitted:
            self.fit(time, rate)
        
        duong_fit = self.fit(time, rate)
        duong_eur = self.compute_eur()
        
        arps_eur = arps_result.get("eur_barrels", 0) if arps_result else 0
        
        eur_difference_pct = (duong_eur["eur_barrels"] - arps_eur) / arps_eur * 100 if arps_eur > 0 else 0
        
        return {
            "duong_eur": duong_eur["eur_barrels"],
            "arps_eur": arps_eur,
            "eur_difference_pct": float(eur_difference_pct),
            "duong_rmse": duong_fit.get("rmse", float('inf')),
            "arps_rmse": arps_result.get("rmse", float('inf')) if arps_result else float('inf'),
            "preferred_model": "duong" if duong_fit.get("rmse", float('inf')) < arps_result.get("rmse", float('inf')) else "arps"
        }
