"""Promotion economics: what each promotion week actually returned.

For every promotion week we ask the elasticity model what that store would have
sold of that product at its regular price with no feature or display: the
baseline. Then:

    incremental units  = actual units - baseline units
    discount given away = (base price - promo price) x baseline units
                          (the money handed to shoppers who would have bought anyway)
    incremental profit  = incremental units x (promo price - unit cost)
                          - discount given away

There is no cost data in the dataset, so unit cost comes from an assumed retail
gross margin (30% by default) and every profit figure is shown at 20%, 30% and
40% so the conclusion can be checked against the assumption.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .elasticity import add_features


def baseline_units(d: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Add model-based baseline units (regular price, no feature/display) to promo weeks."""
    d = add_features(d)
    out = []
    for upc, m in models.items():
        dp = d[d["upc"] == upc].copy()
        smear = np.mean(np.exp(m.resid))  # Duan smearing: unbiased back-transform from logs
        base = dp.assign(log_price=np.log(dp["base_price"]), feature=0, display=0, post_promo=0)
        dp["baseline"] = np.exp(m.predict(base)) * smear
        dp["fitted"] = np.exp(m.predict(dp)) * smear
        out.append(dp)
    return pd.concat(out, ignore_index=True)


def breakeven_lift(discount: float, margin: float) -> float:
    """Unit uplift needed for a price cut to earn the same gross profit as no promotion.

    Profit per unit falls from (margin) to (margin - discount) as a share of base price,
    so volume has to rise by margin / (margin - discount) - 1.
    """
    if discount >= margin:
        return np.inf
    return margin / (margin - discount) - 1


def promo_economics(db: pd.DataFrame, margin: float = 0.30) -> pd.DataFrame:
    p = db[db["on_promo"] == 1].copy()
    cost = p["base_price"] * (1 - margin)
    p["incr_units"] = p["units"] - p["baseline"]
    p["discount_cost"] = (p["base_price"] - p["price"]) * p["baseline"]
    p["incr_revenue"] = p["units"] * p["price"] - p["baseline"] * p["base_price"]
    p["incr_profit"] = p["incr_units"] * (p["price"] - cost) - p["discount_cost"]
    p["lift"] = p["units"] / p["baseline"] - 1
    p["needed_lift"] = [breakeven_lift(x, margin) for x in p["discount"]]
    return p


def summarise(p: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = p.groupby(by)
    s = pd.DataFrame({
        "promo_weeks": g.size(),
        "avg_discount": g["discount"].mean(),
        "avg_lift": g["incr_units"].sum() / g["baseline"].sum(),
        "incremental_share": g["incr_units"].sum() / g["units"].sum(),
        "incr_revenue": g["incr_revenue"].sum(),
        "discount_cost": g["discount_cost"].sum(),
        "incr_profit": g["incr_profit"].sum(),
        "share_profitable": g["incr_profit"].apply(lambda x: (x > 0).mean()),
    })
    s["profit_per_discount_$"] = s["incr_profit"] / s["discount_cost"]
    return s
