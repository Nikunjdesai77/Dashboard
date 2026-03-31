"""
pages/2_Market_Analysis.py — Deep-Dive Market Analysis
"""

import os, sys
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import APP_NAME

from utils.helpers import (
    html, fmt_price, sec_title, page_header,
    chart_km_vs_price, chart_city_price, chart_correlation,
    chart_depreciation, chart_seller_box, chart_transmission_violin,
)

_CSS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "styles.css")
if os.path.exists(_CSS):
    with open(_CSS, encoding="utf-8") as f:
        html(f"<style>\n{f.read()}\n</style>")

dff = st.session_state.get("dff", pd.DataFrame())
if dff.empty:
    st.warning("⚠️ No data. Visit the Home page first.")
    st.stop()

html(page_header(
    "📈 Market Analysis",
    "Price deep-dives · Depreciation curves · Correlation · Segment analysis",
))

tab1, tab2, tab3, tab4 = st.tabs([
    "💰 Price Deep Dive", "📉 Depreciation", "🔗 Correlations", "🏪 Segments",
])

# ── Tab 1 ──────────────────────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        html(sec_title("🛣️", "KM Driven vs Price"))
        st.plotly_chart(chart_km_vs_price(dff), use_container_width=True)
    with c2:
        html(sec_title("🏙️", "City-wise Median Price"))
        st.plotly_chart(chart_city_price(dff), use_container_width=True)

    city_tbl = (
        dff.groupby("city")["selling_price"]
        .agg(Count="count", Avg="mean", Median="median", Min="min", Max="max")
        .sort_values("Median", ascending=False).reset_index()
    )
    for c in ["Avg","Median","Min","Max"]:
        city_tbl[c] = city_tbl[c].apply(fmt_price)
    html(sec_title("🏙️", "City Breakdown Table"))
    st.dataframe(city_tbl, use_container_width=True)

# ── Tab 2 ──────────────────────────────────────────────────────────────────────
with tab2:
    brands_opt = ["All Brands"] + sorted(dff["brand_clean"].unique())
    sel_brand  = st.selectbox("Select brand:", brands_opt)
    brand_arg  = None if sel_brand == "All Brands" else sel_brand
    st.plotly_chart(chart_depreciation(dff, brand_arg), use_container_width=True)

    d_src = dff if brand_arg is None else dff[dff["brand_clean"] == brand_arg]
    dep = (
        d_src.groupby("car_age")["selling_price"]
        .median().reset_index().sort_values("car_age")
    )
    dep.columns = ["Car Age (Yrs)", "Median Price (₹)"]
    if len(dep) > 1:
        base = dep["Median Price (₹)"].iloc[0]
        dep["% Retained"] = (dep["Median Price (₹)"] / base * 100).round(1).astype(str) + "%"
        dep["Median Price (₹)"] = dep["Median Price (₹)"].apply(fmt_price)
    html(sec_title("📄", "Depreciation Table"))
    st.dataframe(dep, use_container_width=True, height=280)

# ── Tab 3 ──────────────────────────────────────────────────────────────────────
with tab3:
    html(sec_title("🔗", "Correlation Matrix"))
    st.plotly_chart(chart_correlation(dff), use_container_width=True)

    num_cols = [c for c in ["car_age","km_driven","owner_num"] if c in dff.columns]
    corr = dff[["selling_price"] + num_cols].corr()["selling_price"].drop("selling_price")
    html(sec_title("📊", "Feature Correlation with Price"))
    for feat, val in corr.sort_values().items():
        lbl  = feat.replace("_"," ").title()
        col  = "#10B981" if val > 0 else "#F43F5E"
        pct  = abs(val) * 100
        dirn = "↑ positive" if val > 0 else "↓ negative"
        html(f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;
                    padding:10px 16px;background:#15213A;
                    border:1px solid rgba(148,163,184,.1);border-radius:10px;">
          <span style="width:160px;font-size:.84rem;color:#94A3B8;">{lbl}</span>
          <div style="flex:1;background:#0D1526;border-radius:4px;height:8px;">
            <div style="width:{pct:.0f}%;height:8px;border-radius:4px;background:{col};"></div>
          </div>
          <span style="color:{col};font-weight:700;font-size:.84rem;width:55px;text-align:right;">{val:.3f}</span>
          <span style="color:#64748B;font-size:.75rem;width:90px;">{dirn}</span>
        </div>""")

# ── Tab 4 ──────────────────────────────────────────────────────────────────────
with tab4:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        html(sec_title("⚙️", "Transmission vs Price"))
        st.plotly_chart(chart_transmission_violin(dff), use_container_width=True)
    with c2:
        html(sec_title("🏪", "Seller Type vs Price"))
        st.plotly_chart(chart_seller_box(dff), use_container_width=True)

    seg = (
        dff.groupby(["fuel","transmission"])["selling_price"]
        .agg(Count="count", Avg="mean", Median="median")
        .sort_values("Median", ascending=False).reset_index()
    )
    seg["Avg"]    = seg["Avg"].apply(fmt_price)
    seg["Median"] = seg["Median"].apply(fmt_price)
    html(sec_title("📋", "Segment Summary"))
    st.dataframe(seg, use_container_width=True)
