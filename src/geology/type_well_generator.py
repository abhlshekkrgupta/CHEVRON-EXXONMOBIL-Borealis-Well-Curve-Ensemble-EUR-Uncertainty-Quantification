import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional, List

class TypeWellGenerator:
    """
    Synthetic type well creation for basin-level EUR estimation.
    
    Generates representative production profiles for a given basin
    and formation by aggregating offset well data and normalizing
    for well design parameters including lateral length, proppant
    loading, and completion design.
    
    Type wells are the fundamental unit of unconventional resource
    assessment. A basin with 5,000 locations at 500,000 bbls EUR
    per well represents 2.5 billion barrels of recoverable resource.
    Getting the type well right determines whether that number is
    2.0 billion or 3.0 billion — a $200 billion swing at $80/bbl.
    """
    
    def __init__(self, basin_name: str, formation: str):
        self.basin_name = basin_name
        self.formation = formation
        self.offset_wells = []
        self.type_well_profile = None
        self.normalization_params = {}
    
    def add_offset_well(
        self,
        well_id: str,
        lateral_length_ft: float,
        proppant_lbs_per_ft: float,
        fluid_bbls_per_ft: float,
        stages: int,
        peak_rate_bbl_day: float,
        eur_barrels: float,
        production_profile: np.ndarray
    ) -> None:
        self.offset_wells.append({
            "well_id": well_id,
            "lateral_length_ft": lateral_length_ft,
            "proppant_lbs_per_ft": proppant_lbs_per_ft,
            "fluid_bbls_per_ft": fluid_bbls_per_ft,
            "stages": stages,
            "peak_rate_bbl_day": peak_rate_bbl_day,
            "eur_barrels": eur_barrels,
            "production_profile": production_profile,
            "eur_per_ft": eur_barrels / max(lateral_length_ft, 1),
            "peak_rate_per_ft": peak_rate_bbl_day / max(lateral_length_ft, 1)
        })
    
    def normalize_to_type_well(
        self,
        target_lateral_length: float = 10000.0,
        target_proppant: float = 2000.0,
        target_fluid: float = 50.0
    ) -> Dict:
        if len(self.offset_wells) < 3:
            return {"status": "insufficient_offset_wells"}
        
        self.normalization_params = {
            "target_lateral_length": target_lateral_length,
            "target_proppant": target_proppant,
            "target_fluid": target_fluid
        }
        
        normalized_eurs = []
        normalized_peak_rates = []
        
        for well in self.offset_wells:
            length_factor = target_lateral_length / max(well["lateral_length_ft"], 1000)
            proppant_factor = (target_proppant / max(well["proppant_lbs_per_ft"], 500)) ** 0.6
            fluid_factor = (target_fluid / max(well["fluid_bbls_per_ft"], 10)) ** 0.3
            
            normalization_multiplier = length_factor * proppant_factor * fluid_factor
            
            normalized_eurs.append(well["eur_barrels"] * normalization_multiplier)
            normalized_peak_rates.append(well["peak_rate_bbl_day"] * normalization_multiplier)
        
        normalized_eurs = np.array(normalized_eurs)
        normalized_peak_rates = np.array(normalized_peak_rates)
        
        if len(self.offset_wells[0]["production_profile"]) > 0:
            profile_length = len(self.offset_wells[0]["production_profile"])
            normalized_profiles = np.zeros((len(self.offset_wells), profile_length))
            
            for i, well in enumerate(self.offset_wells):
                if len(well["production_profile"]) >= profile_length:
                    profile_eur = np.sum(well["production_profile"][:profile_length])
                    if profile_eur > 0:
                        normalized_profiles[i] = well["production_profile"][:profile_length] / profile_eur * normalized_eurs[i]
            
            type_profile = np.mean(normalized_profiles, axis=0)
            type_profile_p10 = np.percentile(normalized_profiles, 10, axis=0)
            type_profile_p90 = np.percentile(normalized_profiles, 90, axis=0)
        else:
            type_profile = np.array([])
            type_profile_p10 = np.array([])
            type_profile_p90 = np.array([])
        
        self.type_well_profile = {
            "mean_eur": float(np.mean(normalized_eurs)),
            "median_eur": float(np.median(normalized_eurs)),
            "p10_eur": float(np.percentile(normalized_eurs, 10)),
            "p50_eur": float(np.percentile(normalized_eurs, 50)),
            "p90_eur": float(np.percentile(normalized_eurs, 90)),
            "std_eur": float(np.std(normalized_eurs)),
            "mean_peak_rate": float(np.mean(normalized_peak_rates)),
            "type_profile": type_profile.tolist() if len(type_profile) > 0 else [],
            "profile_p10": type_profile_p10.tolist() if len(type_profile_p10) > 0 else [],
            "profile_p90": type_profile_p90.tolist() if len(type_profile_p90) > 0 else []
        }
        
        return {
            "basin": self.basin_name,
            "formation": self.formation,
            "n_offset_wells": len(self.offset_wells),
            "normalization_params": self.normalization_params,
            "type_well": self.type_well_profile
        }
    
    def compute_resource_potential(
        self,
        n_locations: int,
        recovery_factor: float = 0.08,
        success_rate: float = 0.95
    ) -> Dict:
        if self.type_well_profile is None:
            return {"status": "type_well_not_generated"}
        
        mean_eur = self.type_well_profile["mean_eur"]
        p10_eur = self.type_well_profile["p10_eur"]
        p90_eur = self.type_well_profile["p90_eur"]
        
        expected_locations = n_locations * success_rate
        
        total_resource_mean = mean_eur * expected_locations
        total_resource_p10 = p10_eur * expected_locations
        total_resource_p90 = p90_eur * expected_locations
        
        return {
            "n_locations": n_locations,
            "success_rate": success_rate,
            "expected_producing_wells": float(expected_locations),
            "total_resource_mean_mmbo": float(total_resource_mean / 1e6),
            "total_resource_p10_mmbo": float(total_resource_p10 / 1e6),
            "total_resource_p90_mmbo": float(total_resource_p90 / 1e6),
            "resource_per_location_mbo": float(mean_eur / 1e3)
        }
