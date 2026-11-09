"""Price elasticity and promotion effects, one model per product.

For each product we fit, across 77 stores and 156 weeks:

    log(units) = store effect + month effect + year effect
                 + elasticity * log(price)
                 + feature + display              (non-price promotion support)
                 + post_promo                     (the week after a promotion: the "dip")

* Store effects compare each store with itself, so a big store selling more
  doesn't look like a price effect.
* Month and year effects take out seasonality and trend.
* Standard errors are clustered by store.

The elasticity is the % change in units for a 1% change in price, holding
promotion support constant. Known limitation: prices in shop data are not set at
random (retailers cut price when they expect demand), so observational
elasticities tend to overstate price sensitivity. Read them as a ranking of
which products respond most, not as exact numbers.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

FORMULA = ("log_units ~ log_price + feature + display + post_promo"
           " + C(store) + C(month) + C(year)")


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values(["upc", "store", "week"]).copy()
    d["log_units"] = np.log(d["units"])
    d["log_price"] = np.log(d["price"])
    d["month"] = d["week"].dt.month
    d["year"] = d["week"].dt.year
    g = d.groupby(["upc", "store"])
    prev_week = g["week"].shift(1)
    prev_promo = g["on_promo"].shift(1)
    consecutive = (d["week"] - prev_week).dt.days == 7
    # Post-promo week: last week was on promotion, this week isn't.
    d["post_promo"] = ((prev_promo == 1) & consecutive & (d["on_promo"] == 0)).astype(int)
    return d


def fit_product(dp: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.ols(FORMULA, data=dp).fit(cov_type="cluster", cov_kwds={"groups": dp["store"]})
    return m


def estimate_all(d: pd.DataFrame, min_rows: int = 500) -> tuple[pd.DataFrame, dict]:
    """Return one row of results per product, plus the fitted models."""
    d = add_features(d)
    rows, models = [], {}
    for upc, dp in d.groupby("upc"):
        if len(dp) < min_rows or dp["log_price"].std() < 0.01:
            continue
        m = fit_product(dp)
        models[upc] = m
        ci = m.conf_int().loc["log_price"]
        first = dp.iloc[0]
        rows.append({
            "upc": upc, "product": f'{first["description"].title()} ({first["product_size"].lower()})', "manufacturer": first["manufacturer"],
            "category": first["category"], "rows": len(dp),
            "elasticity": m.params["log_price"], "ci_low": ci[0], "ci_high": ci[1],
            "feature_lift": np.exp(m.params["feature"]) - 1 if dp["feature"].sum() else np.nan,
            "display_lift": np.exp(m.params["display"]) - 1 if dp["display"].sum() else np.nan,
            "post_promo_dip": np.exp(m.params["post_promo"]) - 1,
            "post_promo_p": m.pvalues["post_promo"],
            "r2": m.rsquared,
            "avg_base_price": dp["base_price"].mean(),
            "weekly_units_per_store": dp["units"].mean(),
        })
    res = pd.DataFrame(rows).sort_values("elasticity").reset_index(drop=True)
    res["plausible"] = res["ci_high"] < 0  # demand should fall when price rises
    return res, models
