"""What-if price changes, using each product's elasticity.

units_new = units * (1 + change) ** elasticity
Margin is assumed (no cost data); revenue doesn't depend on it.
"""
import pandas as pd


def scenario(elasticity: float, change: float, margin: float = 0.30) -> dict:
    vol = (1 + change) ** elasticity - 1
    rev = (1 + change) * (1 + vol) - 1
    # profit per unit as a share of the old price: margin + change
    profit = ((margin + change) * (1 + vol)) / margin - 1
    return {"units": vol, "revenue": rev, "gross_profit": profit}


def price_scenarios(res: pd.DataFrame, changes=(-0.10, -0.05, 0.05, 0.10), margin=0.30) -> pd.DataFrame:
    """Median-elasticity scenarios per category."""
    rows = []
    for cat, e in res.groupby("category")["elasticity"].median().items():
        for c in changes:
            s = scenario(e, c, margin)
            rows.append({"Category": cat, "Elasticity": round(e, 2), "Price change": f"{c:+.0%}",
                         "Units": f"{s['units']:+.1%}", "Revenue": f"{s['revenue']:+.1%}",
                         "Gross profit": f"{s['gross_profit']:+.1%}"})
    return pd.DataFrame(rows)
