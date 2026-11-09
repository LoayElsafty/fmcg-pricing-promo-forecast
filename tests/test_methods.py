"""Method tests on synthetic data with known answers (no dunnhumby file needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fmcg.elasticity import estimate_all  # noqa: E402
from fmcg.forecast import score  # noqa: E402
from fmcg.promo import baseline_units, breakeven_lift, promo_economics  # noqa: E402
from fmcg.simulate import scenario  # noqa: E402


def synthetic_panel(elasticity=-2.0, feature_effect=0.5, seed=0):
    """20 stores x 120 weeks for one product, with a known elasticity and feature lift."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2010-01-06", periods=120, freq="7D")
    rows = []
    for s in range(20):
        size = rng.uniform(0.5, 2.0)
        for w in weeks:
            on = rng.random() < 0.2
            disc = rng.choice([0.1, 0.2, 0.3]) if on else 0.0
            feat = int(on and rng.random() < 0.5)
            price = 3.0 * (1 - disc)
            mu = np.log(40 * size) + elasticity * np.log(price) + feature_effect * feat
            units = max(1, int(round(np.exp(mu + rng.normal(0, 0.15)))))
            rows.append(dict(upc=1, store=s, week=w, units=units, price=price, base_price=3.0,
                             feature=feat, display=0, tpr_only=int(on and not feat), on_promo=int(on),
                             discount=disc, description="Test", manufacturer="M", category="Cat",
                             product_size="1 oz", spend=units * price))
    return pd.DataFrame(rows)


def test_elasticity_recovers_known_value():
    res, _ = estimate_all(synthetic_panel(elasticity=-2.0), min_rows=100)
    row = res.iloc[0]
    assert row["ci_low"] < -2.0 < row["ci_high"]
    assert abs(row["elasticity"] + 2.0) < 0.15
    assert abs(row["feature_lift"] - (np.exp(0.5) - 1)) < 0.1


def test_baseline_strips_promo_effect():
    d = synthetic_panel(elasticity=-2.0, feature_effect=0.0)
    res, models = estimate_all(d, min_rows=100)
    db = baseline_units(d, models)
    promo = db[db["on_promo"] == 1]
    # Units over baseline should match the price effect implied by the discount.
    implied = (1 - promo["discount"]) ** -2.0 - 1
    actual = promo["units"].sum() / promo["baseline"].sum() - 1
    expected = (promo["baseline"] * implied).sum() / promo["baseline"].sum()
    assert abs(actual - expected) < 0.1


@pytest.mark.parametrize("disc,margin,expected", [(0.10, 0.30, 0.5), (0.20, 0.40, 1.0), (0.15, 0.30, 1.0)])
def test_breakeven_lift(disc, margin, expected):
    assert breakeven_lift(disc, margin) == pytest.approx(expected)


def test_breakeven_impossible_when_discount_exceeds_margin():
    assert breakeven_lift(0.35, 0.30) == np.inf


def test_promo_profit_is_zero_at_breakeven():
    p = pd.DataFrame({"on_promo": [1], "units": [150.0], "baseline": [100.0], "price": [0.9],
                      "base_price": [1.0], "discount": [0.10]})
    e = promo_economics(p, margin=0.30)
    assert e["incr_profit"].iloc[0] == pytest.approx(0, abs=1e-9)


def test_scenario_math():
    s = scenario(-2.0, 0.10, margin=0.30)
    assert s["units"] == pytest.approx(1.1 ** -2 - 1)
    assert s["revenue"] == pytest.approx(1.1 ** -1 - 1)
    assert s["gross_profit"] == pytest.approx((0.40 * 1.1 ** -2) / 0.30 - 1)


def test_wape_and_bias():
    bt = pd.DataFrame({"units": [100, 100], "naive": [110, 90], "seasonal_naive": [100, 100],
                       "moving_avg_8": [120, 120], "ml_no_plan": [100, 100], "ml": [100, 80]})
    s = score(bt).iloc[0]
    assert s["wape_naive"] == pytest.approx(0.10)
    assert s["bias_naive"] == pytest.approx(0.0)
    assert s["wape_moving_avg_8"] == pytest.approx(0.20)
    assert s["bias_moving_avg_8"] == pytest.approx(0.20)
    assert s["wape_ml"] == pytest.approx(0.10)
