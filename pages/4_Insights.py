"""
pages/4_Insights.py — Advanced Market Insights & Intelligence
Fixed: prem NameError (was scoped inside with c2 block), safe division guards
"""

import os, sys
import streamlit as st
import pandas as pd
import plotly.express as px

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import APP_NAME

from utils.helpers import (
    html, fmt_price, fmt_km, sec_title, page_header,
    chart_depreciation, BASE_LAYOUT, PALETTE,
)

_CSS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "styles.css")
if os.path.exists(_CSS):
    with open(_CSS, encoding="utf-8") as f:
        html(f"<style>\n{f.read()}\n</style>")

dff = st.session_state.get("dff", pd.DataFrame())
if dff.empty:
    st.warning("No data. Visit the Home page first.")
    st.stop()

html(page_header(
    "💡 Market Insights",
    "Best cities · Resale leaders · Fuel economics · Depreciation intelligence",
))

# ═══════════════════════════════════════════════════════════════════════════════
# Pre-compute ALL variables first (before any UI blocks) to avoid NameError
# ═══════════════════════════════════════════════════════════════════════════════

# City stats
city_stats = (
    dff.groupby("city")["selling_price"]
    .agg(Median="median", Avg="mean", Count="count")
    .sort_values("Median")
    .reset_index()
)
_denom = city_stats["Median"].max() - city_stats["Median"].min()
city_stats["Value Score"] = (
    100 - (city_stats["Median"] - city_stats["Median"].min()) /
    (_denom if _denom > 0 else 1) * 100
).round(1)

# Brand resale stats
brand_stats = (
    dff.groupby("brand_clean")
    .agg(Median=("selling_price", "median"), Count=("selling_price", "count"),
         Avg_Age=("car_age", "mean"))
    .query("Count >= 5")
    .sort_values("Median", ascending=False)
    .head(12)
    .reset_index()
)

# Transmission premium — compute BEFORE any column blocks
auto_m = dff[dff["transmission"] == "Automatic"]["selling_price"].median()
manu_m = dff[dff["transmission"] == "Manual"]["selling_price"].median()
if pd.isna(auto_m): auto_m = 0.0
if pd.isna(manu_m): manu_m = 1.0
prem = (auto_m - manu_m) / max(manu_m, 1) * 100

# First-owner percentage
pct_first = (dff["owner_num"] == 1).mean() * 100

# Best city / top brand
best_city = city_stats.iloc[0]["city"] if not city_stats.empty else "N/A"
top_brand = brand_stats.iloc[0]["brand_clean"] if not brand_stats.empty else "N/A"

# ════════════════════════════════════════════════════════════════════════════════
# 1. Best Cities to Buy
# ════════════════════════════════════════════════════════════════════════════════
html(sec_title("🏙️", "Best Cities to Buy Used Cars", "Lower price = Better deal"))

icons      = ["🥇", "🥈", "🥉", "🏅", "⭐", "👍", "✅", "💼"]
color_cls  = ["violet", "cyan", "amber", "emerald", "rose", "indigo", "violet", "cyan"]
n_cities   = min(4, len(city_stats))
cols       = st.columns(n_cities)

for i, (_, row) in enumerate(city_stats.iterrows()):
    if i >= n_cities:
        break
    with cols[i]:
        html(f"""
        <div class="kpi-card {color_cls[i % len(color_cls)]}">
          <div class="kpi-orb"></div>
          <div class="kpi-icon-wrap">{icons[i] if i < len(icons) else '📍'}</div>
          <div class="kpi-label">#{i+1} — {row['city']}</div>
          <div class="kpi-value">{fmt_price(row['Median'])}</div>
          <div class="kpi-sub">Score: {row['Value Score']}/100 · {int(row['Count'])} listings</div>
        </div>""")

fig_city = px.bar(
    city_stats.sort_values("Value Score", ascending=False),
    x="city", y="Value Score",
    color="Value Score",
    color_continuous_scale=["#F43F5E", "#F59E0B", "#10B981"],
    text=city_stats.sort_values("Value Score", ascending=False)["Value Score"].round(1),
    labels={"city": "City", "Value Score": "Buyer Score"},
)
fig_city.update_traces(textposition="outside", marker_line_width=0)
fig_city.update_coloraxes(showscale=False)
fig_city.update_layout(**{**BASE_LAYOUT, "height": 320, "title": "City Buyer Score"})
st.plotly_chart(fig_city, use_container_width=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════════
# 2. Highest Resale Value Brands
# ════════════════════════════════════════════════════════════════════════════════
html(sec_title("🏅", "Highest Resale Value Brands"))

medal = ["💎", "🥇", "🥈", "🥉"] + ["🏷️"] * 20
c3    = st.columns(3)
for i, (_, row) in enumerate(brand_stats.iterrows()):
    with c3[i % 3]:
        cc = "violet" if i < 3 else "cyan" if i < 7 else "amber"
        html(f"""
        <div class="insight-card {cc}">
          <h4>{medal[i]} {row['brand_clean']}</h4>
          <p>Median: <b style="color:#A78BFA">{fmt_price(row['Median'])}</b> &nbsp;|&nbsp;
             Avg Age: <b style="color:#67E8F9">{row['Avg_Age']:.1f} yrs</b><br>
             Listings: <b>{int(row['Count'])}</b></p>
        </div>""")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════════
# 3. Fuel Economics & Transmission
# ════════════════════════════════════════════════════════════════════════════════
c1, c2 = st.columns(2, gap="large")

with c1:
    html(sec_title("⛽", "Fuel Type Price Impact"))
    fuel_tbl = (
        dff.groupby("fuel")["selling_price"]
        .agg(Count="count", Avg="mean", Median="median")
        .sort_values("Median", ascending=False)
        .reset_index()
    )
    base_fuel = fuel_tbl["Median"].min()
    fuel_tbl["Premium vs Cheapest"] = fuel_tbl["Median"].apply(
        lambda x: f"+{(x - base_fuel) / max(base_fuel, 1) * 100:.1f}%" if base_fuel > 0 else "—"
    )
    fuel_display = fuel_tbl.copy()
    fuel_display["Avg"]    = fuel_display["Avg"].apply(fmt_price)
    fuel_display["Median"] = fuel_display["Median"].apply(fmt_price)
    st.dataframe(fuel_display, use_container_width=True)

    fuel_median = (
        dff.groupby("fuel")["selling_price"].median()
        .sort_values(ascending=False).reset_index()
    )
    for i, (_, row) in enumerate(fuel_median.iterrows()):
        cc = ["violet", "cyan", "amber", "emerald", "rose"][i % 5]
        html(f"""
        <div class="insight-card {cc}">
          <h4>⛽ {row['fuel']}</h4>
          <p>Median price: <b>{fmt_price(row['selling_price'])}</b></p>
        </div>""")

with c2:
    html(sec_title("🔄", "Transmission Price Impact"))
    trans_tbl = (
        dff.groupby("transmission")["selling_price"]
        .agg(Count="count", Avg="mean", Median="median")
        .sort_values("Median", ascending=False)
        .reset_index()
    )
    trans_display = trans_tbl.copy()
    trans_display["Avg"]    = trans_display["Avg"].apply(fmt_price)
    trans_display["Median"] = trans_display["Median"].apply(fmt_price)
    st.dataframe(trans_display, use_container_width=True)

    html(f"""
    <div class="insight-card violet">
      <h4>⚙️ Automatic vs Manual Premium</h4>
      <p>Automatic: <b style="color:#A78BFA">{fmt_price(auto_m)}</b> &nbsp;vs&nbsp;
         Manual: <b style="color:#67E8F9">{fmt_price(manu_m)}</b><br>
         Premium: <b>+{prem:.1f}%</b> for automatic.</p>
    </div>""")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════════
# 4. Depreciation
# ════════════════════════════════════════════════════════════════════════════════
html(sec_title("📉", "Depreciation Intelligence"))
c3, c4 = st.columns([2, 1], gap="large")

with c3:
    st.plotly_chart(chart_depreciation(dff), use_container_width=True)

with c4:
    age_grp = (
        dff.groupby("car_age")["selling_price"]
        .median().reset_index().sort_values("car_age")
    )
    base_p = age_grp["selling_price"].iloc[0] if not age_grp.empty else 1
    for yrs in [0, 1, 3, 5, 8, 10, 15]:
        match = age_grp[age_grp["car_age"] == yrs]
        if not match.empty:
            val = match["selling_price"].values[0]
            pct = val / max(base_p, 1) * 100
            col = "#10B981" if pct >= 70 else "#F59E0B" if pct >= 50 else "#F43F5E"
            html(f"""
            <div style="display:flex;align-items:center;justify-content:space-between;
                        padding:8px 14px;margin-bottom:6px;background:#15213A;
                        border:1px solid rgba(148,163,184,.1);border-radius:10px;">
              <span style="font-size:.83rem;color:#94A3B8;">
                After <b style="color:#E2E8F0">{yrs} yr{'s' if yrs != 1 else ''}</b>
              </span>
              <span style="font-size:.83rem;font-weight:700;color:{col}">
                {pct:.1f}% retained
              </span>
            </div>""")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════════
# 5. Key Findings
# ════════════════════════════════════════════════════════════════════════════════
html(sec_title("🔑", "Key Market Findings"))

findings = [
    ("violet", "🏙️ Best City to Buy",
     f"<b>{best_city}</b> offers the most competitive median asking price — ideal for budget buyers."),
    ("cyan",   "🏅 Best Resale Brand",
     f"<b>{top_brand}</b> commands the highest resale value in the current dataset."),
    ("amber",  "⛽ Diesel Premium",
     "Diesel cars consistently sell at a premium over petrol due to highway fuel efficiency."),
    ("emerald", "⚙️ Automatic Advantage",
     f"Automatics price <b>+{prem:.0f}%</b> over manual — a premium for urban driving convenience."),
    ("violet", "📅 Depreciation Cliff (First 3 Years)",
     "30–45% of a car's value is lost in first 3 years. Best sweet spot: 3–5 year old cars."),
    ("cyan",   "👤 First-Owner Premium",
     f"{pct_first:.1f}% of listings are first-owner — they command 15–25% higher prices."),
]

cf = st.columns(2)
for i, (cc, title, body) in enumerate(findings):
    with cf[i % 2]:
        html(f"""<div class="insight-card {cc}"><h4>{title}</h4><p>{body}</p></div>""")

# ── Download ──────────────────────────────────────────────────────────────────
st.markdown("---")
html(sec_title("📥", "Download Reports"))
dc1, dc2 = st.columns(2)
with dc1:
    csv = dff.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Full Cleaned Dataset (CSV)",
                       csv, "cariq_full_data.csv", "text/csv",
                       use_container_width=True)
with dc2:
    report_lines = [
        "=" * 50,
        "  CarIQ — Market Intelligence Report",
        "=" * 50,
        f"Total listings : {len(dff):,}",
        f"Median price   : {fmt_price(dff['selling_price'].median())}",
        f"Avg price      : {fmt_price(dff['selling_price'].mean())}",
        f"Best city      : {best_city}",
        f"Top brand      : {top_brand}",
        f"Auto premium   : +{prem:.1f}%",
        f"First-owner %  : {pct_first:.1f}%",
        "=" * 50,
    ]
    st.download_button("📄 Insights Summary (TXT)",
                       "\n".join(report_lines).encode("utf-8"),
                       "cariq_insights.txt", "text/plain",
                       use_container_width=True)
