"""Load and clean dunnhumby's "Breakfast at the Frat" data.

156 weeks (Jan 2009 - Jan 2012) of weekly sales for 55 products in 77 US stores,
across four categories (cold cereal, frozen pizza, pretzels, mouthwash), with
shelf price, base (regular) price and three promotion flags:
FEATURE (in the weekly ad), DISPLAY (in-store display), TPR_ONLY (price cut only).

The raw workbook is dunnhumby's and is not redistributed here: download it free
from https://www.dunnhumby.com/source-files/ and put it in data/raw/.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "dunnhumby - Breakfast at the Frat.xlsx"
CACHE = ROOT / "data" / "panel.parquet"


def _read_raw():
    tx = pd.read_excel(RAW, "dh Transaction Data", header=1)
    pr = pd.read_excel(RAW, "dh Products Lookup", header=1)
    st = pd.read_excel(RAW, "dh Store Lookup", header=1)
    return tx, pr, st


def build_panel(tx: pd.DataFrame, pr: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """One row per store x product x week, cleaned and labelled."""
    d = tx.rename(columns=str.lower).rename(columns={"week_end_date": "week", "store_num": "store"})
    d["week"] = pd.to_datetime(d["week"])
    n0 = len(d)
    # Clean: drop rows without a price, and rows that sold nothing or were priced at zero.
    d = d.dropna(subset=["price", "base_price"])
    d = d[(d["units"] > 0) & (d["price"] > 0) & (d["base_price"] > 0)]
    # Shelf price above base price is a recording artefact: cap it at base price.
    d["price"] = np.minimum(d["price"], d["base_price"])
    d["discount"] = 1 - d["price"] / d["base_price"]
    for c in ("feature", "display", "tpr_only"):
        d[c] = d[c].astype(int)
    d["on_promo"] = ((d["feature"] + d["display"] + d["tpr_only"]) > 0).astype(int)
    d["mechanic"] = np.select(
        [(d.feature == 1) & (d.display == 1), d.feature == 1, d.display == 1, d.tpr_only == 1],
        ["Feature + display", "Feature", "Display", "Price cut only"], default="No promotion")
    p = pr.rename(columns=str.lower)[["upc", "description", "manufacturer", "category", "sub_category", "product_size"]]
    s = st.rename(columns=str.lower).rename(columns={"store_id": "store", "seg_value_name": "store_segment",
                                                     "address_state_prov_code": "state"})[["store", "store_segment", "state"]]
    # Two stores (4503, 17627) are listed twice with different segments; keep the first listing.
    s = s.drop_duplicates("store", keep="first")
    p = p.drop_duplicates("upc", keep="first")
    n1 = len(d)
    d = d.merge(p, on="upc", how="left").merge(s, on="store", how="left")
    assert len(d) == n1, "lookup joins must not duplicate rows"
    d["category"] = d["category"].str.title()
    d.attrs["rows_dropped"] = n0 - len(d)
    return d.sort_values(["upc", "store", "week"]).reset_index(drop=True)


def load_panel(rebuild: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not rebuild:
        return pd.read_parquet(CACHE)
    d = build_panel(*_read_raw())
    d.to_parquet(CACHE)
    return d
