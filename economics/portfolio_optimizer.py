import numpy as np
from typing import Dict, List, Tuple, Optional

class PortfolioOptimizer:
    """
    Well portfolio optimization under capital constraints.
    
    Allocates a constrained drilling budget across candidate wells
    to maximize expected portfolio NPV while managing risk through
    diversification across basins, well types, and geological plays.
    """
    
    def __init__(self, annual_budget: float = 500000000):
        self.annual_budget = annual_budget
        self.wells = []
        self.portfolio = None
    
    def add_well_candidate(
        self,
        well_id: str,
        basin: str,
        drill_cost: float,
        expected_eur: float,
        eur_std: float,
        expected_npv: float,
        npv_std: float,
        success_probability: float = 1.0,
        correlation_group: str = "default"
    ) -> None:
        self.wells.append({
            "well_id": well_id,
            "basin": basin,
            "drill_cost": drill_cost,
            "expected_eur": expected_eur,
            "eur_std": eur_std,
            "expected_npv": expected_npv,
            "npv_std": npv_std,
            "success_probability": success_probability,
            "correlation_group": correlation_group,
            "npv_index": expected_npv / max(drill_cost, 1),
            "risk_adjusted_npv": expected_npv * success_probability
        })
    
    def greedy_portfolio_selection(self) -> Dict:
        sorted_wells = sorted(
            self.wells,
            key=lambda w: w["risk_adjusted_npv"] / max(w["drill_cost"], 1),
            reverse=True
        )
        
        selected = []
        remaining_budget = self.annual_budget
        total_expected_npv = 0.0
        total_expected_eur = 0.0
        
        for well in sorted_wells:
            if well["drill_cost"] <= remaining_budget:
                selected.append(well)
                remaining_budget -= well["drill_cost"]
                total_expected_npv += well["expected_npv"] * well["success_probability"]
                total_expected_eur += well["expected_eur"] * well["success_probability"]
        
        self.portfolio = selected
        
        return {
            "n_wells_selected": len(selected),
            "total_capital_deployed": float(self.annual_budget - remaining_budget),
            "budget_utilization_pct": float((self.annual_budget - remaining_budget) / self.annual_budget * 100),
            "remaining_budget": float(remaining_budget),
            "expected_portfolio_npv": float(total_expected_npv),
            "expected_portfolio_eur": float(total_expected_eur),
            "portfolio_npv_index": float(total_expected_npv / max(self.annual_budget - remaining_budget, 1)),
            "selected_wells": [w["well_id"] for w in selected]
        }
    
    def simulate_portfolio_outcomes(self, n_simulations: int = 10000) -> Dict:
        if self.portfolio is None:
            self.greedy_portfolio_selection()
        
        portfolio_npvs = np.zeros(n_simulations)
        portfolio_eurs = np.zeros(n_simulations)
        
        correlation_matrix = self._build_correlation_matrix()
        
        try:
            L = np.linalg.cholesky(correlation_matrix)
        except np.linalg.LinAlgError:
            eigenvalues = np.linalg.eigvalsh(correlation_matrix)
            min_eig = np.min(eigenvalues)
            if min_eig < 0:
                correlation_matrix += np.eye(len(self.portfolio)) * (abs(min_eig) + 0.01)
            L = np.linalg.cholesky(correlation_matrix)
        
        for sim in range(n_simulations):
            z = np.random.randn(len(self.portfolio))
            correlated_z = L @ z
            
            portfolio_npv = 0.0
            portfolio_eur = 0.0
            
            for i, well in enumerate(self.portfolio):
                npv_sim = well["expected_npv"] + well["npv_std"] * correlated_z[i]
                eur_sim = well["expected_eur"] + well["eur_std"] * correlated_z[i]
                
                if np.random.random() < well["success_probability"]:
                    portfolio_npv += npv_sim
                    portfolio_eur += eur_sim
                else:
                    portfolio_npv -= well["drill_cost"] * 0.5
                    portfolio_eur += 0
            
            portfolio_npvs[sim] = portfolio_npv
            portfolio_eurs[sim] = portfolio_eur
        
        return {
            "expected_npv": float(np.mean(portfolio_npvs)),
            "npv_p10": float(np.percentile(portfolio_npvs, 10)),
            "npv_p50": float(np.percentile(portfolio_npvs, 50)),
            "npv_p90": float(np.percentile(portfolio_npvs, 90)),
            "npv_std": float(np.std(portfolio_npvs)),
            "probability_positive_npv": float(np.mean(portfolio_npvs > 0)),
            "expected_eur": float(np.mean(portfolio_eurs)),
            "eur_p10": float(np.percentile(portfolio_eurs, 10)),
            "eur_p90": float(np.percentile(portfolio_eurs, 90)),
            "n_simulations": n_simulations
        }
    
    def _build_correlation_matrix(self) -> np.ndarray:
        n = len(self.portfolio)
        corr = np.eye(n)
        
        for i in range(n):
            for j in range(i+1, n):
                if self.portfolio[i]["basin"] == self.portfolio[j]["basin"]:
                    corr[i, j] = 0.6
                elif self.portfolio[i]["correlation_group"] == self.portfolio[j]["correlation_group"]:
                    corr[i, j] = 0.4
                else:
                    corr[i, j] = 0.1
                corr[j, i] = corr[i, j]
        
        return corr
    
    def efficient_frontier(
        self,
        budget_range: List[float]
    ) -> Dict:
        frontier = []
        original_budget = self.annual_budget
        
        for budget in budget_range:
            self.annual_budget = budget
            selection = self.greedy_portfolio_selection()
            
            if selection["n_wells_selected"] > 0:
                simulation = self.simulate_portfolio_outcomes(n_simulations=5000)
                
                frontier.append({
                    "budget": budget,
                    "n_wells": selection["n_wells_selected"],
                    "expected_npv": simulation["expected_npv"],
                    "npv_std": simulation["npv_std"],
                    "sharpe_ratio": simulation["expected_npv"] / max(simulation["npv_std"], 1)
                })
        
        self.annual_budget = original_budget
        
        return {
            "frontier": frontier,
            "budget_range": budget_range,
            "optimal_budget": max(frontier, key=lambda x: x["sharpe_ratio"])["budget"] if frontier else None
        }
