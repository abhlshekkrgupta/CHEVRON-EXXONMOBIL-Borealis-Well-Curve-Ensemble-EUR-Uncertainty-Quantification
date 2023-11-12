import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional, List, Callable

class ConformalPrediction:
    """
    Distribution-free uncertainty quantification via conformal prediction.
    
    Unlike bootstrap methods that require distributional assumptions,
    conformal prediction provides valid prediction intervals under
    the sole assumption of exchangeability. Produces calibrated
    uncertainty bands for decline curve forecasts without assuming
    Gaussian errors or any specific error distribution.
    
    Implements split conformal prediction with a calibration set
    to produce prediction intervals with guaranteed coverage.
    """
    
    def __init__(self, confidence_level: float = 0.90):
        self.confidence_level = confidence_level
        self.calibration_scores = None
        self.q_hat = None
    
    def split_conformal(
        self,
        time_train: np.ndarray,
        rate_train: np.ndarray,
        time_cal: np.ndarray,
        rate_cal: np.ndarray,
        fitting_fn: Callable,
        prediction_fn: Callable
    ) -> Dict:
        fit_result = fitting_fn(time_train, rate_train)
        
        cal_predictions = prediction_fn(fit_result, time_cal)
        
        residuals = np.abs(rate_cal - cal_predictions)
        self.calibration_scores = residuals
        
        n_cal = len(residuals)
        q_index = int(np.ceil((n_cal + 1) * self.confidence_level))
        q_index = min(q_index, n_cal - 1)
        
        self.q_hat = np.sort(residuals)[q_index]
        
        coverage = np.mean(residuals <= self.q_hat)
        
        return {
            "q_hat": float(self.q_hat),
            "empirical_coverage": float(coverage),
            "target_coverage": self.confidence_level,
            "n_calibration_points": n_cal,
            "valid": coverage >= self.confidence_level - 0.05
        }
    
    def predict_with_intervals(
        self,
        time_forecast: np.ndarray,
        point_predictions: np.ndarray
    ) -> Dict:
        if self.q_hat is None:
            raise ValueError("Must call split_conformal first")
        
        lower_bound = point_predictions - self.q_hat
        upper_bound = point_predictions + self.q_hat
        
        lower_bound = np.maximum(lower_bound, 0)
        
        return {
            "point_predictions": point_predictions.tolist(),
            "lower_bound": lower_bound.tolist(),
            "upper_bound": upper_bound.tolist(),
            "prediction_interval_width": float(np.mean(upper_bound - lower_bound)),
            "q_hat": float(self.q_hat)
        }
    
    def adaptive_conformal(
        self,
        time: np.ndarray,
        rate: np.ndarray,
        fitting_fn: Callable,
        prediction_fn: Callable,
        window_size: int = 30
    ) -> Dict:
        n = len(time)
        
        if n < window_size * 2:
            return {"status": "insufficient_data"}
        
        train_end = n - window_size
        
        time_train = time[:train_end]
        rate_train = rate[:train_end]
        time_cal = time[train_end:]
        rate_cal = rate[train_end:]
        
        result = self.split_conformal(
            time_train, rate_train, time_cal, rate_cal,
            fitting_fn, prediction_fn
        )
        
        forecast_time = np.linspace(time[-1], time[-1] * 2, 100)
        fit = fitting_fn(time_train, rate_train)
        forecasts = prediction_fn(fit, forecast_time)
        
        intervals = self.predict_with_intervals(forecast_time, forecasts)
        
        return {
            **result,
            "forecast_intervals": intervals,
            "window_size": window_size,
            "n_train": len(time_train),
            "n_calibration": len(time_cal)
        }
    
    def compute_eur_conformal(
        self,
        eur_point_estimate: float,
        eur_ensemble: np.ndarray
    ) -> Dict:
        residuals = np.abs(eur_ensemble - eur_point_estimate)
        
        q_index = int(np.ceil((len(residuals) + 1) * self.confidence_level))
        q_index = min(q_index, len(residuals) - 1)
        q = np.sort(residuals)[q_index]
        
        return {
            "eur_point_estimate": float(eur_point_estimate),
            "eur_lower_bound": float(max(0, eur_point_estimate - q)),
            "eur_upper_bound": float(eur_point_estimate + q),
            "interval_width": float(2 * q),
            "relative_width_pct": float(2 * q / eur_point_estimate * 100) if eur_point_estimate > 0 else 0,
            "confidence_level": self.confidence_level
        }
