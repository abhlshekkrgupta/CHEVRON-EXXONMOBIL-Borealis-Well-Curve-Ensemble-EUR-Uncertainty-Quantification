import numpy as np
from typing import Dict, Tuple, Optional, List

class NPVCalculator:
    """
    Net present value calculator for oil and gas wells under uncertainty.
    
    Computes the discounted cash flow of a well's production stream
    accounting for oil price forecasts, operating costs, royalties,
    taxes, and time value of money. Integrates with EUR uncertainty
    to produce probabilistic NPV estimates.
    """
    
    def __init__(self, discount_rate: float = 0.10):
        self.discount_rate = discount_rate
        self.price_forecast = None
        self.cost_structure = None
    
    def set_price_forecast(
        self,
        oil_price_per_bbl: float,
        gas_price_per_mcf: float,
        ngl_price_per_bbl: float = 25.0,
        escalation_rate: float = 0.02
    ) -> Dict:
        self.price_forecast = {
            "oil": oil_price_per_bbl,
            "gas": gas_price_per_mcf,
            "ngl": ngl_price_per_bbl,
            "escalation_rate": escalation_rate
        }
        return self.price_forecast
    
    def set_cost_structure(
        self,
        fixed_opex_per_month: float = 5000.0,
        variable_opex_per_bbl: float = 3.50,
        royalty_rate: float = 0.20,
        severance_tax_rate: float = 0.05,
        ad_valorem_tax_rate: float = 0.02
    ) -> Dict:
        self.cost_structure = {
            "fixed_opex_monthly": fixed_opex_per_month,
            "variable_opex_per_bbl": variable_opex_per_bbl,
            "royalty_rate": royalty_rate,
            "severance_tax_rate": severance_tax_rate,
            "ad_valorem_tax_rate": ad_valorem_tax_rate
        }
        return self.cost_structure
    
    def compute_annual_cash_flows(
        self,
        annual_production: np.ndarray,
        oil_cut: float = 0.75,
        gas_cut: float = 0.15,
        ngl_cut: float = 0.10
    ) -> Dict:
        if self.price_forecast is None or self.cost_structure is None:
            raise ValueError("Must set price forecast and cost structure")
        
        n_years = len(annual_production)
        cash_flows = np.zeros(n_years)
        
        for year in range(n_years):
            escalated_oil = self.price_forecast["oil"] * (1 + self.price_forecast["escalation_rate"]) ** year
            escalated_gas = self.price_forecast["gas"] * (1 + self.price_forecast["escalation_rate"]) ** year
            escalated_ngl = self.price_forecast["ngl"] * (1 + self.price_forecast["escalation_rate"]) ** year
            
            annual_oil = annual_production[year] * oil_cut
            annual_gas = annual_production[year] * gas_cut
            annual_ngl = annual_production[year] * ngl_cut
            
            gross_revenue = (
                annual_oil * escalated_oil +
                annual_gas * escalated_gas / 6.0 +
                annual_ngl * escalated_ngl
            )
            
            royalties = gross_revenue * self.cost_structure["royalty_rate"]
            
            net_revenue = gross_revenue - royalties
            
            operating_cost = (
                self.cost_structure["fixed_opex_monthly"] * 12 +
                annual_production[year] * self.cost_structure["variable_opex_per_bbl"]
            )
            
            severance_tax = gross_revenue * self.cost_structure["severance_tax_rate"]
            ad_valorem_tax = gross_revenue * self.cost_structure["ad_valorem_tax_rate"]
            
            cash_flow = net_revenue - operating_cost - severance_tax - ad_valorem_tax
            cash_flows[year] = cash_flow
        
        return {
            "annual_cash_flows": cash_flows.tolist(),
            "total_undiscounted_cash_flow": float(np.sum(cash_flows)),
            "peak_cash_flow_year": int(np.argmax(cash_flows)),
            "n_positive_years": int(np.sum(cash_flows > 0))
        }
    
    def compute_npv(
        self,
        annual_production: np.ndarray,
        initial_investment: float,
        oil_cut: float = 0.75,
        gas_cut: float = 0.15,
        ngl_cut: float = 0.10
    ) -> Dict:
        cash_flows = self.compute_annual_cash_flows(annual_production, oil_cut, gas_cut, ngl_cut)
        
        annual_cf = np.array(cash_flows["annual_cash_flows"])
        
        discount_factors = (1 + self.discount_rate) ** (-np.arange(len(annual_cf)))
        
        discounted_cf = annual_cf * discount_factors
        npv = np.sum(discounted_cf) - initial_investment
        
        cumulative_cf = np.cumsum(annual_cf) - initial_investment
        payback_mask = cumulative_cf >= 0
        payback_year = np.argmax(payback_mask) if np.any(payback_mask) else None
        
        if npv > 0 and np.sum(discounted_cf[discounted_cf > 0]) > 0:
            profitability_index = (npv + initial_investment) / initial_investment
        else:
            profitability_index = 0.0
        
        return {
            "npv": float(npv),
            "initial_investment": initial_investment,
            "total_discounted_cash_flow": float(np.sum(discounted_cf)),
            "profitability_index": float(profitability_index),
            "payback_year": payback_year,
            "discount_rate": self.discount_rate,
            "annual_cash_flows": cash_flows["annual_cash_flows"],
            "discounted_cash_flows": discounted_cf.tolist()
        }
    
    def compute_probabilistic_npv(
        self,
        eur_samples: np.ndarray,
        initial_investment: float,
        well_life_years: int = 30
    ) -> Dict:
        n_samples = len(eur_samples)
        npv_samples = np.zeros(n_samples)
        
        for i in range(n_samples):
            annual_prod = self._eur_to_annual_production(eur_samples[i], well_life_years)
            
            cash_flows = self.compute_annual_cash_flows(annual_prod)
            annual_cf = np.array(cash_flows["annual_cash_flows"])
            discount_factors = (1 + self.discount_rate) ** (-np.arange(len(annual_cf)))
            npv_samples[i] = np.sum(annual_cf * discount_factors) - initial_investment
        
        return {
            "npv_mean": float(np.mean(npv_samples)),
            "npv_median": float(np.median(npv_samples)),
            "npv_p10": float(np.percentile(npv_samples, 10)),
            "npv_p50": float(np.percentile(npv_samples, 50)),
            "npv_p90": float(np.percentile(npv_samples, 90)),
            "npv_std": float(np.std(npv_samples)),
            "probability_positive_npv": float(np.mean(npv_samples > 0)),
            "expected_monetary_value": float(np.mean(npv_samples)),
            "n_samples": n_samples
        }
    
    def _eur_to_annual_production(self, eur: float, well_life_years: int) -> np.ndarray:
        total = eur
        
        decline_curve = np.exp(-0.3 * np.arange(well_life_years))
        decline_curve = decline_curve / np.sum(decline_curve)
        
        return total * decline_curve
