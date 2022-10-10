import numpy as np
from typing import Dict, Tuple, Optional, List

class BreakevenAnalysis:
    """
    Breakeven oil price calculator for well economics.
    
    Determines the minimum oil price required for a well to achieve
    a target rate of return, accounting for all costs, royalties,
    taxes, and production decline. Used for go/no-go drilling
    decisions and portfolio prioritization.
    """
    
    def __init__(self, target_irr: float = 0.10):
        self.target_irr = target_irr
        self.price_sensitivity = None
    
    def compute_breakeven_price(
        self,
        eur_barrels: float,
        initial_investment: float,
        well_life_years: int = 30,
        fixed_opex_monthly: float = 5000,
        variable_opex_per_bbl: float = 3.50,
        royalty_rate: float = 0.20,
        severance_tax_rate: float = 0.05,
        discount_rate: float = 0.10
    ) -> Dict:
        annual_production = eur_barrels * np.exp(-0.3 * np.arange(well_life_years))
        annual_production = annual_production / np.sum(annual_production) * eur_barrels
        
        lo, hi = 10.0, 200.0
        
        for _ in range(100):
            mid = (lo + hi) / 2
            
            npv = self._compute_npv_at_price(
                mid, annual_production, initial_investment,
                fixed_opex_monthly, variable_opex_per_bbl,
                royalty_rate, severance_tax_rate, discount_rate
            )
            
            if npv > 0:
                hi = mid
            else:
                lo = mid
        
        breakeven_price = (lo + hi) / 2
        
        npv_at_breakeven = self._compute_npv_at_price(
            breakeven_price, annual_production, initial_investment,
            fixed_opex_monthly, variable_opex_per_bbl,
            royalty_rate, severance_tax_rate, discount_rate
        )
        
        return {
            "breakeven_price_per_bbl": float(breakeven_price),
            "eur_barrels": float(eur_barrels),
            "initial_investment": initial_investment,
            "npv_at_breakeven": float(npv_at_breakeven),
            "breakeven_price_per_flowing_barrel": float(breakeven_price / (eur_barrels / well_life_years)) if eur_barrels > 0 else 0,
            "target_irr": self.target_irr
        }
    
    def _compute_npv_at_price(
        self,
        oil_price: float,
        annual_production: np.ndarray,
        initial_investment: float,
        fixed_opex: float,
        variable_opex: float,
        royalty_rate: float,
        severance_tax_rate: float,
        discount_rate: float
    ) -> float:
        n_years = len(annual_production)
        npv = -initial_investment
        
        for year in range(n_years):
            gross_revenue = annual_production[year] * oil_price
            royalties = gross_revenue * royalty_rate
            opex = fixed_opex * 12 + annual_production[year] * variable_opex
            severance = gross_revenue * severance_tax_rate
            
            cash_flow = gross_revenue - royalties - opex - severance
            npv += cash_flow / (1 + discount_rate) ** (year + 1)
        
        return npv
    
    def compute_price_sensitivity(
        self,
        eur_barrels: float,
        initial_investment: float,
        price_range: Tuple[float, float] = (30.0, 120.0),
        n_prices: int = 20
    ) -> Dict:
        prices = np.linspace(price_range[0], price_range[1], n_prices)
        annual_production = eur_barrels * np.exp(-0.3 * np.arange(30))
        annual_production = annual_production / np.sum(annual_production) * eur_barrels
        
        npvs = np.zeros(n_prices)
        for i, price in enumerate(prices):
            npvs[i] = self._compute_npv_at_price(
                price, annual_production, initial_investment,
                5000, 3.50, 0.20, 0.05, 0.10
            )
        
        positive_mask = npvs > 0
        if np.any(positive_mask):
            breakeven_idx = np.argmax(positive_mask)
            breakeven_price = prices[breakeven_idx]
        else:
            breakeven_price = prices[-1]
        
        self.price_sensitivity = {
            "prices": prices.tolist(),
            "npvs": npvs.tolist(),
            "breakeven_price": float(breakeven_price)
        }
        
        return {
            "breakeven_price": float(breakeven_price),
            "price_sensitivity": self.price_sensitivity,
            "npv_at_50_dollars": float(np.interp(50.0, prices, npvs)),
            "npv_at_80_dollars": float(np.interp(80.0, prices, npvs)),
            "npv_at_100_dollars": float(np.interp(100.0, prices, npvs))
        }
    
    def compare_well_economics(
        self,
        wells: List[Dict]
    ) -> Dict:
        results = []
        
        for well in wells:
            breakeven = self.compute_breakeven_price(
                eur_barrels=well.get("eur", 500000),
                initial_investment=well.get("drill_cost", 8000000),
                well_life_years=well.get("life_years", 30)
            )
            
            results.append({
                "well_name": well.get("name", "unknown"),
                "basin": well.get("basin", "unknown"),
                "eur_barrels": well.get("eur", 0),
                "breakeven_price": breakeven["breakeven_price_per_bbl"],
                "drill_cost": well.get("drill_cost", 0),
                "finding_cost_per_bbl": well.get("drill_cost", 0) / max(well.get("eur", 1), 1)
            })
        
        results.sort(key=lambda x: x["breakeven_price"])
        
        return {
            "well_results": results,
            "n_wells": len(results),
            "lowest_breakeven": results[0] if results else None,
            "highest_breakeven": results[-1] if results else None,
            "median_breakeven": results[len(results)//2]["breakeven_price"] if results else 0
        }
