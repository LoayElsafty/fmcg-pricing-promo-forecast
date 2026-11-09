"""Weekly demand forecasts by product, tested honestly.

Grain: total units per product per week across all stores (55 series x 156 weeks).

Backtest (rolling origin): stand at a week in the last year, train only on the
weeks before it, forecast the next 4 weeks, score, move forward 4 weeks, repeat.
13 origins x 4 weeks x 55 products. Nothing after the origin is used except
the promotion plan and planned shelf prices, which a supplier knows in advance.

Models compared:
  * naive            - next weeks = last week's sales
  * seasonal naive   - same week last year
  * moving average   - mean of the last 8 weeks
  * gradient boosting (ML) - one model across all products: recent sales level,
    seasonality, planned price vs regular price, and the share of stores
    running a feature, display or price cut that week

Accuracy is WAPE (sum of absolute errors / sum of actual units: lower is better)
and bias (sum of errors / sum of actuals: + means over-forecasting).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HORIZON = 4
N_ORIGINS = 13


def weekly_series(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(["upc", "week"])
    w = pd.DataFrame({
        "units": g["units"].sum(),
        "stores": g["store"].nunique(),
        "price": g["spend"].sum() / g["units"].sum(),
        "base_price": g["base_price"].mean(),
        "feature_share": g["feature"].mean(),
        "display_share": g["display"].mean(),
        "tpr_share": g["tpr_only"].mean(),
    }).reset_index()
    w["discount"] = (1 - w["price"] / w["base_price"]).clip(lower=0)
    cats = d.drop_duplicates("upc").set_index("upc")["category"]
    w["category"] = w["upc"].map(cats)
    # Complete grid so lags are on true weeks; weeks with no sales are rare and set to 0.
    weeks = pd.Index(sorted(w["week"].unique()), name="week")
    full = pd.MultiIndex.from_product([sorted(w["upc"].unique()), weeks], names=["upc", "week"])
    w = w.set_index(["upc", "week"]).reindex(full).reset_index()
    w["units"] = w["units"].fillna(0)
    for c in ["feature_share", "display_share", "tpr_share", "discount"]:
        w[c] = w[c].fillna(0)
    w["category"] = w["upc"].map(cats)
    # Keep each product's selling life only (some launch late or are delisted).
    sold = w[w["units"] > 0].groupby("upc")["week"].agg(["min", "max"])
    w = w.join(sold, on="upc")
    w = w[(w["week"] >= w["min"]) & (w["week"] <= w["max"])].drop(columns=["min", "max"]).reset_index(drop=True)
    w["week_of_year"] = w["week"].dt.isocalendar().week.astype(int)
    w["t"] = w.groupby("upc").cumcount()
    return w


def _features(w: pd.DataFrame, origin_t: int) -> pd.DataFrame:
    """Features for every row, using only sales up to origin_t for the level terms."""
    hist = w[w["t"] <= origin_t]
    lvl = hist.groupby("upc").apply(
        lambda g: pd.Series({
            "level_4": g["units"].tail(4).mean(),
            "level_13": g["units"].tail(13).mean(),
            "nonpromo_level": g.loc[g["feature_share"] + g["display_share"] + g["tpr_share"] < 0.1, "units"].tail(13).mean(),
        }), include_groups=False)
    f = w.join(lvl, on="upc")
    f["nonpromo_level"] = f["nonpromo_level"].fillna(f["level_13"])
    return f


FEATURES = ["level_4", "level_13", "nonpromo_level", "discount", "feature_share", "display_share",
            "tpr_share", "week_of_year", "cat_code"]


def _train_rows(w: pd.DataFrame, origin_t: int) -> pd.DataFrame:
    """Training examples that mimic forecasting: level known 1-4 weeks before the target week."""
    rows = []
    for o in range(max(26, origin_t - 78), origin_t - HORIZON + 1, 2):
        f = _features(w, o)
        rows.append(f[(f["t"] > o) & (f["t"] <= o + HORIZON)])
    return pd.concat(rows)


NO_PLAN = [c for c in FEATURES if c not in ("discount", "feature_share", "display_share", "tpr_share")]


def backtest(w: pd.DataFrame) -> pd.DataFrame:
    w = w.copy()
    w["cat_code"] = w["category"].astype("category").cat.codes
    last_t = w["t"].max()
    origins = [last_t - HORIZON * k for k in range(N_ORIGINS, 0, -1)]
    out = []
    for o in origins:
        train = _train_rows(w, o)
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                              l2_regularization=1.0, random_state=0)
        model.fit(train[FEATURES], np.log1p(train["units"]))
        # Same model without the promotion plan, to measure what the plan is worth.
        blind = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                              l2_regularization=1.0, random_state=0)
        blind.fit(train[NO_PLAN], np.log1p(train["units"]))
        f = _features(w, o)
        test = f[(f["t"] > o) & (f["t"] <= o + HORIZON)].copy()
        test["origin"] = o
        test["ml"] = np.expm1(model.predict(test[FEATURES])).clip(min=0)
        test["ml_no_plan"] = np.expm1(blind.predict(test[NO_PLAN])).clip(min=0)
        hist = w[w["t"] <= o].set_index(["upc", "t"])["units"]
        test["naive"] = [hist.get((u, o), np.nan) for u in test["upc"]]
        test["seasonal_naive"] = [hist.get((u, t - 52), np.nan) for u, t in zip(test["upc"], test["t"])]
        test["moving_avg_8"] = test["upc"].map(w[(w["t"] <= o) & (w["t"] > o - 8)].groupby("upc")["units"].mean())
        out.append(test)
    return pd.concat(out, ignore_index=True)


MODELS = ["naive", "seasonal_naive", "moving_avg_8", "ml_no_plan", "ml"]
LABELS = {"naive": "Last week", "seasonal_naive": "Same week last year",
          "moving_avg_8": "8-week average", "ml_no_plan": "Gradient boosting, no promo plan", "ml": "Gradient boosting (price + promo plan)"}


def score(bt: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    # Score every model on the same rows: drop weeks where any model has no forecast
    # (e.g. no "same week last year" for a product launched less than a year earlier).
    bt = bt.dropna(subset=MODELS)
    def s(g):
        r = {}
        for m in MODELS:
            e = g[m] - g["units"]
            r[f"wape_{m}"] = e.abs().sum() / g["units"].sum()
            r[f"bias_{m}"] = e.sum() / g["units"].sum()
        return pd.Series(r)
    if by:
        return bt.groupby(by).apply(s, include_groups=False)
    return s(bt).to_frame("all").T
