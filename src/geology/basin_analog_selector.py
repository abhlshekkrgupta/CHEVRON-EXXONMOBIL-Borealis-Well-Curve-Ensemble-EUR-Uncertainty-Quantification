import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional

class BasinAnalogSelector:
    """
    Analog well selection for new basin evaluation.
    
    When entering a new basin with limited production history,
    selects analog wells from mature basins with similar geological
    characteristics to inform EUR estimates. Uses multi-dimensional
    matching on depth, thickness, porosity, permeability, pressure,
    and thermal maturity.
    """
    
    def __init__(self):
        self.mature_basin_database = {}
        self.geological_features = [
            "depth_ft", "thickness_ft", "porosity_pct",
            "permeability_md", "pressure_psi", "thermal_maturity_ro",
            "clay_content_pct", "organic_richness_toc"
        ]
    
    def register_mature_basin(
        self,
        basin_name: str,
        formation: str,
        geological_params: Dict,
        eur_statistics: Dict,
        n_wells: int
    ) -> None:
        key = f"{basin_name}_{formation}"
        self.mature_basin_database[key] = {
            "basin": basin_name,
            "formation": formation,
            "geological_params": geological_params,
            "eur_statistics": eur_statistics,
            "n_wells": n_wells
        }
    
    def find_analogs(
        self,
        target_geology: Dict,
        top_n: int = 5,
        feature_weights: Optional[Dict[str, float]] = None
    ) -> Dict:
        if len(self.mature_basin_database) < 1:
            return {"status": "no_mature_basins_registered"}
        
        if feature_weights is None:
            feature_weights = {
                "depth_ft": 0.15,
                "thickness_ft": 0.15,
                "porosity_pct": 0.20,
                "permeability_md": 0.15,
                "pressure_psi": 0.10,
                "thermal_maturity_ro": 0.10,
                "clay_content_pct": 0.05,
                "organic_richness_toc": 0.10
            }
        
        target_vector = []
        feature_list = []
        for feature in self.geological_features:
            if feature in target_geology and feature in feature_weights:
                target_vector.append(target_geology[feature])
                feature_list.append(feature)
        
        target_vector = np.array(target_vector)
        
        basin_names = []
        basin_vectors = []
        
        for key, basin_data in self.mature_basin_database.items():
            geo = basin_data["geological_params"]
            vector = []
            for feature in feature_list:
                if feature in geo:
                    vector.append(geo[feature])
                else:
                    vector.append(0)
            
            basin_names.append(key)
            basin_vectors.append(vector)
        
        basin_vectors = np.array(basin_vectors)
        
        scaler_std = np.std(basin_vectors, axis=0)
        scaler_std[scaler_std < 1e-6] = 1.0
        
        target_norm = target_vector / scaler_std
        basin_norm = basin_vectors / scaler_std
        
        weights_array = np.array([feature_weights.get(f, 0.1) for f in feature_list])
        weights_array = weights_array / np.sum(weights_array)
        
        weighted_distances = np.zeros(len(basin_names))
        for i in range(len(basin_names)):
            diff = (target_norm - basin_norm[i]) * np.sqrt(weights_array)
            weighted_distances[i] = np.sqrt(np.sum(diff ** 2))
        
        sorted_indices = np.argsort(weighted_distances)
        
        analogs = []
        for rank, idx in enumerate(sorted_indices[:top_n]):
            key = basin_names[idx]
            basin_data = self.mature_basin_database[key]
            
            similarity_score = np.exp(-weighted_distances[idx])
            
            analogs.append({
                "rank": rank + 1,
                "basin": basin_data["basin"],
                "formation": basin_data["formation"],
                "similarity_score": float(similarity_score),
                "distance": float(weighted_distances[idx]),
                "eur_statistics": basin_data["eur_statistics"],
                "n_wells": basin_data["n_wells"],
                "geological_params": basin_data["geological_params"]
            })
        
        weights = np.array([a["similarity_score"] for a in analogs])
        weights = weights / np.sum(weights)
        
        weighted_eur_mean = np.sum([a["eur_statistics"].get("mean_eur", 0) * w for a, w in zip(analogs, weights)])
        weighted_eur_p10 = np.sum([a["eur_statistics"].get("p10_eur", 0) * w for a, w in zip(analogs, weights)])
        weighted_eur_p90 = np.sum([a["eur_statistics"].get("p90_eur", 0) * w for a, w in zip(analogs, weights)])
        
        return {
            "target_geology": target_geology,
            "analogs": analogs,
            "n_analogs": len(analogs),
            "weighted_eur_mean": float(weighted_eur_mean),
            "weighted_eur_p10": float(weighted_eur_p10),
            "weighted_eur_p90": float(weighted_eur_p90),
            "eur_uncertainty_ratio": float((weighted_eur_p90 - weighted_eur_p10) / max(weighted_eur_mean, 1))
        }
    
    def cross_validate_analogs(
        self,
        target_basin: str,
        target_formation: str,
        n_folds: int = 5
    ) -> Dict:
        if len(self.mature_basin_database) < n_folds + 3:
            return {"status": "insufficient_data_for_cross_validation"}
        
        all_basins = list(self.mature_basin_database.keys())
        
        if f"{target_basin}_{target_formation}" in all_basins:
            all_basins.remove(f"{target_basin}_{target_formation}")
        
        np.random.shuffle(all_basins)
        
        fold_errors = []
        
        for fold in range(min(n_folds, len(all_basins))):
            test_key = all_basins[fold]
            train_keys = [k for k in all_basins if k != test_key]
            
            original_db = self.mature_basin_database.copy()
            self.mature_basin_database = {k: original_db[k] for k in train_keys}
            
            test_geology = original_db[test_key]["geological_params"]
            test_actual_eur = original_db[test_key]["eur_statistics"]["mean_eur"]
            
            result = self.find_analogs(test_geology, top_n=5)
            predicted_eur = result.get("weighted_eur_mean", 0)
            
            if test_actual_eur > 0:
                error_pct = (predicted_eur - test_actual_eur) / test_actual_eur * 100
                fold_errors.append(error_pct)
            
            self.mature_basin_database = original_db
        
        return {
            "n_folds": len(fold_errors),
            "mean_absolute_error_pct": float(np.mean(np.abs(fold_errors))) if fold_errors else 0,
            "mean_error_pct": float(np.mean(fold_errors)) if fold_errors else 0,
            "std_error_pct": float(np.std(fold_errors)) if fold_errors else 0,
            "fold_errors": [float(e) for e in fold_errors]
        }
