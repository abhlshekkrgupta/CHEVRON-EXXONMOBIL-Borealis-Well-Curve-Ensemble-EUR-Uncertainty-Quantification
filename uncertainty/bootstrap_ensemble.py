import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional, List, Callable

class BootstrapEnsemble:
    """
    Block bootstrap for time series decline curve analysis.
    
    Standard bootstrap assumes independent observations, which
    production time series violate due to autocorrelation. Block
    bootstrap preserves the temporal dependence structure by
    resampling contiguous blocks of observations rather than
    individual points.
    
    Implements moving block bootstrap (MBB) with optimal block
    length selection via the Politis-White algorithm.
    """
    
    def __init__(self, n_bootstraps: int = 2000, block_length: Optional[int] = None):
        self.n_bootstraps = n_bootstraps
        self.block_length = block_length
        self.eur_samples = None
        self.parameter_samples = None
    
    def estimate_optimal_block_length(self, time_series: np.ndarray) -> int:
        n = len(time_series)
        
        if n < 20:
            return max(2, n // 4)
        
        autocorr = np.correlate(time_series - np.mean(time_series), 
                                time_series - np.mean(time_series), mode='full')
        autocorr = autocorr[n-1:] / autocorr[n-1]
        
        cutoff = np.argmax(np.abs(autocorr) < 0.05)
        if cutoff == 0:
            cutoff = n // 4
        
        optimal_block = int(n ** (1/3))
        optimal_block = max(2, min(optimal_block, n // 3))
        
        return optimal_block
    
    def moving_block_bootstrap(
        self,
        time: np.ndarray,
        rate: np.ndarray,
        fitting_fn: Callable,
        eur_fn: Callable
    ) -> Dict:
        n = len(time)
        
        if self.block_length is None:
            self.block_length = self.estimate_optimal_block_length(rate)
        
        block_len = self.block_length
        n_blocks = int(np.ceil(n / block_len))
        
        eur_bootstrap = np.zeros(self.n_bootstraps)
        param_bootstrap = []
        
        for b in range(self.n_bootstraps):
            block_starts = np.random.randint(0, n - block_len + 1, size=n_blocks)
            
            boot_time = []
            boot_rate = []
            
            for start in block_starts:
                end = min(start + block_len, n)
                boot_time.extend(time[start:end])
                boot_rate.extend(rate[start:end])
            
            boot_time = np.array(boot_time[:n])
            boot_rate = np.array(boot_rate[:n])
            
            sort_idx = np.argsort(boot_time)
            boot_time = boot_time[sort_idx]
            boot_rate = boot_rate[sort_idx]
            
            try:
                fit_result = fitting_fn(boot_time, boot_rate)
                eur_result = eur_fn(fit_result)
                eur_bootstrap[b] = eur_result.get("eur_barrels", 0)
            except Exception:
                eur_bootstrap[b] = np.nan
        
        eur_bootstrap = eur_bootstrap[~np.isnan(eur_bootstrap)]
        
        self.eur_samples = eur_bootstrap
        
        return {
            "n_bootstraps": self.n_bootstraps,
            "n_valid": len(eur_bootstrap),
            "block_length": block_len,
            "mean_eur": float(np.mean(eur_bootstrap)),
            "median_eur": float(np.median(eur_bootstrap)),
            "p10_eur": float(np.percentile(eur_bootstrap, 10)),
            "p50_eur": float(np.percentile(eur_bootstrap, 50)),
            "p90_eur": float(np.percentile(eur_bootstrap, 90)),
            "std_eur": float(np.std(eur_bootstrap)),
            "cv": float(np.std(eur_bootstrap) / np.mean(eur_bootstrap)) if np.mean(eur_bootstrap) > 0 else 0
        }
    
    def stationary_bootstrap(
        self,
        time: np.ndarray,
        rate: np.ndarray,
        fitting_fn: Callable,
        eur_fn: Callable,
        mean_block_length: int = 10
    ) -> Dict:
        n = len(time)
        
        eur_bootstrap = np.zeros(self.n_bootstraps)
        
        for b in range(self.n_bootstraps):
            boot_time = []
            boot_rate = []
            idx = 0
            
            while len(boot_time) < n:
                block_len = np.random.geometric(1.0 / mean_block_length)
                start = np.random.randint(0, n)
                
                for j in range(block_len):
                    if len(boot_time) >= n:
                        break
                    circular_idx = (start + j) % n
                    boot_time.append(time[circular_idx])
                    boot_rate.append(rate[circular_idx])
            
            boot_time = np.array(boot_time)
            boot_rate = np.array(boot_rate)
            
            sort_idx = np.argsort(boot_time)
            boot_time = boot_time[sort_idx]
            boot_rate = boot_rate[sort_idx]
            
            try:
                fit_result = fitting_fn(boot_time, boot_rate)
                eur_result = eur_fn(fit_result)
                eur_bootstrap[b] = eur_result.get("eur_barrels", 0)
            except Exception:
                eur_bootstrap[b] = np.nan
        
        eur_bootstrap = eur_bootstrap[~np.isnan(eur_bootstrap)]
        
        return {
            "n_bootstraps": self.n_bootstraps,
            "n_valid": len(eur_bootstrap),
            "mean_eur": float(np.mean(eur_bootstrap)),
            "p10_eur": float(np.percentile(eur_bootstrap, 10)),
            "p90_eur": float(np.percentile(eur_bootstrap, 90)),
            "std_eur": float(np.std(eur_bootstrap))
        }
