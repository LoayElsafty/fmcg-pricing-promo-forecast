"""Run the whole analysis and write reports/results.md and reports/figures/*.png.

    python run_all.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fmcg.data import load_panel  # noqa: E402
from fmcg.elasticity import estimate_all  # noqa: E402
from fmcg.forecast import LABELS, MODELS, backtest, score, weekly_series  # noqa: E402
from fmcg.promo import baseline_units, breakeven_lift, promo_economics, summarise  # noqa: E402
from fmcg.simulate import price_scenarios  # noqa: E402

OUT = Path("reports")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Validated categorical palette (slots 1-4), one colour per category, fixed order.
CATS = ["Cold Cereal", "Frozen Pizza", "Bag Snacks", "Oral Hygiene Products"]
COL = dict(zip(CATS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "savefig.dpi": 150,
})


def pct(x, d=0):
    return f"{x * 100:.{d}f}%"


def main():
    d = load_panel()
    res, models = estimate_all(d)
    db = baseline_units(d, models)

    # ---------- Figure 1: elasticity by product ----------
    r = res.copy()
    r["cat_order"] = r["category"].map({c: i for i, c in enumerate(CATS)})
    r = r.sort_values(["cat_order", "elasticity"], ascending=[True, False]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 13))
    y = np.arange(len(r))[::-1]
    for c in CATS:
        m = r["category"] == c
        ax.hlines(y[m], r.loc[m, "ci_low"], r.loc[m, "ci_high"], color=COL[c], lw=2)
        ax.plot(r.loc[m, "elasticity"], y[m], "o", ms=6, color=COL[c], mec=SURF, mew=1.5, label=c)
    ax.set_yticks(y, r["product"], fontsize=7.5)
    ax.axvline(-1, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("Price elasticity: % change in units for a 1% price rise (95% interval)\nleft of the dashed line = elastic, right = inelastic")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Frozen pizza reacts most to price, mouthwash least", x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "elasticity_by_product.png")
    plt.close(fig)

    # ---------- Promo economics ----------
    econ = {m: promo_economics(db, m) for m in (0.20, 0.30, 0.40)}
    p30 = econ[0.30]
    by_mech = {m: summarise(e, ["mechanic"]) for m, e in econ.items()}
    by_cat_mech = summarise(p30, ["category", "mechanic"]).reset_index()

    # Figure 2: lift achieved vs lift needed to break even
    fig, ax = plt.subplots(figsize=(8, 5.2))
    xs = np.linspace(0.02, 0.29, 100)
    for mg, ls in [(0.30, "-"), (0.40, "--")]:
        ax.plot(xs * 100, [breakeven_lift(x, mg) * 100 for x in xs], color=MUTED, lw=1.5, ls=ls)
        ax.text(xs[-1] * 100 - 0.2, breakeven_lift(xs[-1], mg) * 100 if mg == 0.4 else 700,
                f"break-even at {int(mg * 100)}% margin", fontsize=8, color=MUTED, ha="right", va="bottom")
    markers = {"Price cut only": "o", "Display": "s", "Feature": "^", "Feature + display": "D"}
    for _, row in by_cat_mech.iterrows():
        ax.plot(row["avg_discount"] * 100, row["avg_lift"] * 100, markers[row["mechanic"]], ms=8,
                color=COL[row["category"]], mec=SURF, mew=1.5)
    for c in CATS:
        ax.plot([], [], "o", color=COL[c], label=c)
    for k, mk in markers.items():
        ax.plot([], [], mk, color=MUTED, label=k)
    ax.set_ylim(0, 700)
    ax.set_xlabel("Average discount (%)")
    ax.set_ylabel("Extra units vs baseline (%)")
    ax.grid(color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    ax.set_title("Most promotions sit below the line they need to clear", loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "promo_lift_vs_breakeven.png")
    plt.close(fig)

    # Figure 3: profit per $1 of discount, by mechanic, at three margins
    order = ["Price cut only", "Display", "Feature", "Feature + display"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    shades = {0.20: "#a9c8ef", 0.30: "#2a78d6", 0.40: "#164f99"}
    wdt = 0.26
    for i, (mg, s) in enumerate(by_mech.items()):
        vals = s.loc[order, "profit_per_discount_$"]
        ax.bar(np.arange(4) + (i - 1) * wdt, vals, wdt - 0.03, color=shades[mg], label=f"{int(mg * 100)}% margin")
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(np.arange(4), order)
    ax.set_ylabel("Incremental profit per $1 of discount")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=8, title="Assumed retail margin", title_fontsize=8)
    ax.set_title("A price cut on its own loses money at any plausible margin", loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "promo_profit_by_mechanic.png")
    plt.close(fig)

    # ---------- Price simulator ----------
    sims = price_scenarios(res, changes=(-0.10, -0.05, 0.05, 0.10), margin=0.30)

    # ---------- Forecast ----------
    w = weekly_series(d)
    bt = backtest(w)
    sc = score(bt)
    sc_cat = score(bt, ["category"])

    fig, ax = plt.subplots(figsize=(8, 3.6))
    vals = [sc[f"wape_{m}"].iloc[0] * 100 for m in MODELS]
    colors = [GRID] * (len(MODELS) - 1) + ["#2a78d6"]
    ax.barh([LABELS[m] for m in MODELS], vals, color=colors, height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.6, i, f"{v:.1f}%", va="center", fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Forecast error, WAPE (lower is better)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_title("Feeding the promotion plan into the forecast halves the error", loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "forecast_error_by_model.png")
    plt.close(fig)

    ex_upc = res.loc[res["product"].str.startswith("Kell Froot Loops"), "upc"].iloc[0]
    e = bt[bt["upc"] == ex_upc].sort_values("week")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    promo_wk = e[(e["feature_share"] + e["display_share"]) >= 0.3]
    for wk in promo_wk["week"]:
        ax.axvspan(wk - pd.Timedelta(days=3), wk + pd.Timedelta(days=3), color="#eef3fb", lw=0)
    ax.plot(e["week"], e["units"], color=INK, lw=2, label="Actual units")
    ax.plot(e["week"], e["moving_avg_8"], color=MUTED, lw=2, ls="--", label="8-week average")
    ax.plot(e["week"], e["ml"], color="#2a78d6", lw=2, label="Gradient boosting with promo plan")
    ax.set_ylabel("Units per week, all stores")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Kellogg's Froot Loops, last 52 weeks (shaded = feature/display weeks)", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "forecast_example.png")
    plt.close(fig)

    # ---------- results.md ----------
    m30 = by_mech[0.30]
    tpr = m30.loc["Price cut only"]
    fd = m30.loc["Feature + display"]
    tpr_share = m30.loc["Price cut only", "promo_weeks"] / m30["promo_weeks"].sum()
    cat_el = res.groupby("category")["elasticity"].median()
    best_base = min(sc[f"wape_{m}"].iloc[0] for m in ["naive", "seasonal_naive", "moving_avg_8"])
    lines = [
        "# Results", "",
        f"Data: dunnhumby Breakfast at the Frat · {d['week'].min():%b %Y} – {d['week'].max():%b %Y} · "
        f"{d['store'].nunique()} stores · {d['upc'].nunique()} products · {len(d):,} store-product-weeks · "
        f"{int(d['units'].sum()):,} units · ${d['spend'].sum():,.0f} sales.", "",
        "## 1. Price elasticity", "",
        f"All {len(res)} products have a negative elasticity whose 95% interval excludes zero. Median by category:", "",
        "| Category | Median elasticity |", "|---|---|",
        *[f"| {c} | {cat_el[c]:.2f} |" for c in CATS], "",
        "![Elasticity by product](figures/elasticity_by_product.png)", "",
        "## 2. Promotion economics (assumed 30% retail margin; 20% and 40% shown for sensitivity)", "",
        "| Mechanic | Promo weeks | Avg discount | Extra units | Share of promo units that are extra | Profit per $1 of discount (20% / 30% / 40%) | Weeks profitable at 30% |",
        "|---|---|---|---|---|---|---|",
    ]
    for mech in order:
        a = by_mech[0.30].loc[mech]
        lines.append(
            f"| {mech} | {int(a['promo_weeks']):,} | {pct(a['avg_discount'])} | +{pct(a['avg_lift'])} | {pct(a['incremental_share'])} | "
            f"{by_mech[0.20].loc[mech, 'profit_per_discount_$']:.2f} / {a['profit_per_discount_$']:.2f} / {by_mech[0.40].loc[mech, 'profit_per_discount_$']:.2f} | "
            f"{pct(a['share_profitable'])} |")
    lines += ["", "![Lift vs break-even](figures/promo_lift_vs_breakeven.png)", "",
              "![Profit by mechanic](figures/promo_profit_by_mechanic.png)", "",
              "## 3. Price scenarios (30% margin, elasticity held constant)", "",
              sims.to_markdown(index=False), "",
              "## 4. Forecast backtest", "",
              f"{bt['origin'].nunique()} forecast origins over the last year, 4 weeks ahead each, {bt['upc'].nunique()} products, {len(bt):,} product-weeks scored.", "",
              "| Model | WAPE | Bias |", "|---|---|---|",
              *[f"| {LABELS[m]} | {pct(sc[f'wape_{m}'].iloc[0], 1)} | {sc[f'bias_{m}'].iloc[0] * 100:+.1f}% |" for m in MODELS], "",
              "| Category | " + " | ".join(LABELS[m] for m in MODELS) + " |", "|---|" + "---|" * len(MODELS),
              *[f"| {c} | " + " | ".join(pct(sc_cat.loc[c, f'wape_{m}'], 1) for m in MODELS) + " |" for c in CATS], "",
              "![Forecast error](figures/forecast_error_by_model.png)", "",
              "![Forecast example](figures/forecast_example.png)", "",
              "## Headline numbers (used in the README)", "",
              f"- Price-cut-only promotions: {pct(tpr_share)} of promo weeks, +{pct(tpr['avg_lift'])} units on a {pct(tpr['avg_discount'])} discount; "
              f"{pct(1 - tpr['incremental_share'])} of the units sold on deal would have sold anyway; profitable in {pct(tpr['share_profitable'])} of weeks at 30% margin.",
              f"- Feature + display: +{pct(fd['avg_lift'])} units on a {pct(fd['avg_discount'])} discount.",
              f"- Forecast: WAPE {pct(sc['wape_ml'].iloc[0], 1)} vs {pct(best_base, 1)} for the best simple baseline "
              f"({pct(1 - sc['wape_ml'].iloc[0] / best_base)} less error); without the promo plan the same model scores {pct(sc['wape_ml_no_plan'].iloc[0], 1)}.",
              ]
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    res.round(4).to_csv(OUT / "elasticity_by_product.csv", index=False)
    print("\n".join(lines[-4:]))


if __name__ == "__main__":
    main()
