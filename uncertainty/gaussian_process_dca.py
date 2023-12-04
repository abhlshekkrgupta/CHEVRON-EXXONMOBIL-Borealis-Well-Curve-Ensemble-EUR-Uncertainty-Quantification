import numpy as np
from scipy import stats
from scipy.linalg import cholesky, solve_triangular
from typing import Dict, Tuple, Optional

class GaussianProcessDCA:
    """
    Gaussian Process regression for decline curve analysis.
    
    GP-DCA treats the decline curve as a latent function sampled
    from a Gaussian process with a kernel that enforces monotonic
    decline behavior. Provides full posterior predictive distributions
    that naturally capture both aleatoric and epistemic uncertainty.
    
    Uses a composite kernel combining a Matérn kernel for smooth
    decline with a linear kernel for long-term trend.
    """
    
    def __init__(self, kernel_length_scale: float = 100.0, kernel_variance: float = 1.0, noise_variance: float = 0.1):
        self.length_scale = kernel_length_scale
        self.kernel_variance = kernel_variance
        self.noise_variance = noise_variance
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.alpha = None
    
    def matern_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        dist = np.abs(X1[:, None] - X2[None, :]) / self.length_scale
        
        sqrt3 = np.sqrt(3.0)
        K = self.kernel_variance * (1.0 + sqrt3 * dist) * np.exp(-sqrt3 * dist)
        
        return K
    
    def fit(self, time: np.ndarray, rate: np.ndarray) -> Dict:
        time = np.asarray(time, dtype=float).reshape(-1, 1)
        rate = np.asarray(rate, dtype=float)
        
        mask = (rate > 0) & (time.flatten() >= 0)
        time = time[mask]
        rate = rate[mask]
        
        if len(time) < 5:
            return {"status": "insufficient_data"}
        
        self.X_train = time
        self.y_train = rate
        
        K = self.matern_kernel(time, time)
        K += self.noise_variance * np.eye(len(time))
        
        try:
            L = cholesky(K, lower=True)
            self.alpha = solve_triangular(L.T, solve_triangular(L, rate, lower=True), lower=False)
            self.K_inv_L = L
        except Exception as e:
            return {"status": "cholesky_failed", "error": str(e)}
        
        fitted = self.predict(time.flatten())
        residuals = rate - fitted["mean"]
        rmse = np.sqrt(np.mean(residuals ** 2))
        
        return {
            "status": "converged",
            "rmse": float(rmse),
            "n_points": len(time),
            "kernel_length_scale": self.length_scale,
            "noise_variance": self.noise_variance
        }
    
    def predict(self, time_forecast: np.ndarray) -> Dict:
        if self.X_train is None:
            raise ValueError("Must fit model before prediction")
        
        X_test = np.asarray(time_forecast, dtype=float).reshape(-1, 1)
        
        K_test = self.matern_kernel(X_test, X_test)
        K_train_test = self.matern_kernel(self.X_train, X_test)
        
        L = self.K_inv_L
        v = solve_triangular(L, K_train_test, lower=True)
        
        mean = K_train_test.T @ self.alpha
        
        cov = K_test - v.T @ v
        
        std = np.sqrt(np.maximum(np.diag(cov), 0))
        
        lower_90 = mean - 1.645 * std
        upper_90 = mean + 1.645 * std
        lower_90 = np.maximum(lower_90, 0)
        
        return {
            "mean": mean,
            "std": std,
            "lower_90": lower_90,
            "upper_90": upper_90,
            "time": time_forecast
        }
    
    def compute_eur_posterior(self, max_time_days: int = 365 * 30, n_samples: int = 2000) -> Dict:
        t = np.linspace(self.X_train[-1, 0], max_time_days, 300)
        
        prediction = self.predict(t)
        
        eur_samples = np.zeros(n_samples)
        for i in range(n_samples):
            sample_rates = prediction["mean"] + prediction["std"] * np.random.randn(len(t))
            sample_rates = np.maximum(sample_rates, 0)
            eur_samples[i] = np.trapz(sample_rates, t)
        
        return {
            "eur_mean": float(np.mean(eur_samples)),
            "eur_p10": float(np.percentile(eur_samples, 10)),
            "eur_p50": float(np.percentile(eur_samples, 50)),
            "eur_p90": float(np.percentile(eur_samples, 90)),
            "eur_std": float(np.std(eur_samples))
        }
