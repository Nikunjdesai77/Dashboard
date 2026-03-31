"""
pages/1_Overview.py — Dashboard Overview
"""

import os, sys
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import APP_NAME

from utils.helpers import (
    html, fmt_price, fmt_num, fmt_km,
    kpi_card, kpi_row, sec_title, page_header,
    chart_brand_price, chart_fuel_donut,
    chart_price_trend, chart_brand_volume,
    chart_price_distribution,
)

# ── CSS ────────────────────────────────────────────────────────────────────────
_CSS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "styles.css")
if os.path.exists(_CSS):
    with open(_CSS, encoding="utf-8") as f:
        html(f"<style>\n{f.read()}\n</style>")

# ── Data from session_state ────────────────────────────────────────────────────
dff = st.session_state.get("dff", pd.DataFrame())
df  = st.session_state.get("df",  pd.DataFrame())

if dff.empty:
    st.warning("⚠️ No data loaded yet. Please go to the **Home** page first for initialization.")
    st.stop()

# ── KPIs ──────────────────────────────────────────────────────────────────────
total      = len(dff)
avg_price  = dff["selling_price"].mean()
med_price  = dff["selling_price"].median()
top_brand  = dff["brand_clean"].value_counts().idxmax()
avg_km     = dff["km_driven"].mean()
top_fuel   = dff["fuel"].value_counts().idxmax()
pct_first  = (dff["owner_num"] == 1).sum() / total * 100
yr_min     = int(dff["year"].min())
yr_max     = int(dff["year"].max())
n_brands   = dff["brand_clean"].nunique()
n_cities   = dff["city"].nunique()

html(page_header(
    "📊 Dashboard Overview",
    "KPIs · Brand analysis · Price distribution · Data explorer",
    stats=[
        (fmt_num(total),      "Listings"),
        (fmt_price(med_price),"Median"),
        (f"{n_brands}",       "Brands"),
        (f"{n_cities}",       "Cities"),
    ],
))

html(kpi_row([
    kpi_card("📋", "Total Listings",  fmt_num(total),       f"{n_cities} cities",        "violet", 50),
    kpi_card("💰", "Average Price",   fmt_price(avg_price), f"Median {fmt_price(med_price)}","cyan",100),
    kpi_card("🏎️", "Top Brand",       top_brand,            "Most listed brand",         "amber", 150),
    kpi_card("🛣️", "Avg KM Driven",   fmt_km(avg_km),       "Odometer average",          "emerald",200),
]))
html(kpi_row([
    kpi_card("⛽", "Top Fuel",         top_fuel,             "Dominant fuel variant",     "cyan",  250),
    kpi_card("👤", "First-Owner %",   f"{pct_first:.1f}%",   "Of filtered listings",      "violet",300),
    kpi_card("📅", "Year Range",       f"{yr_min}–{yr_max}", "Manufacturing span",        "amber", 350),
    kpi_card("📊", "Brands",           str(n_brands),        "Unique car brands",         "emerald",400),
]))

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🏠 Overview", "📈 Trends", "📊 Distribution"])

with tab1:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        html(sec_title("🏷️", "Brand Price Leaders", f"Top {min(15,n_brands)}"))
        st.plotly_chart(chart_brand_price(dff), use_container_width=True)
    with c2:
        html(sec_title("⛽", "Fuel Type Breakdown"))
        st.plotly_chart(chart_fuel_donut(dff), use_container_width=True)

    html(sec_title("📋", "Summary Statistics"))
    st.dataframe(
        dff["selling_price"].describe().to_frame("Selling Price (₹)").T.round(0).astype(int),
        use_container_width=True,
    )

with tab2:
    html(sec_title("📈", "Price Trend Over Years"))
    st.plotly_chart(chart_price_trend(dff), use_container_width=True)
    html(sec_title("📊", "Listing Volume by Brand"))
    st.plotly_chart(chart_brand_volume(dff), use_container_width=True)

with tab3:
    html(sec_title("📊", "Price Distribution"))
    st.plotly_chart(chart_price_distribution(dff), use_container_width=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        grp = dff.groupby("fuel")["selling_price"].agg(Count="count",Mean="mean",Median="median")
        grp["Mean"]   = grp["Mean"].apply(fmt_price)
        grp["Median"] = grp["Median"].apply(fmt_price)
        html(sec_title("⛽", "By Fuel"))
        st.dataframe(grp.reset_index(), use_container_width=True)
    with c2:
        grp2 = dff.groupby("owner")["selling_price"].agg(Count="count",Median="median")
        grp2["Median"] = grp2["Median"].apply(fmt_price)
        html(sec_title("👤", "By Ownership"))
        st.dataframe(grp2.reset_index(), use_container_width=True)

# ── Download ──────────────────────────────────────────────────────────────────
st.markdown("---")
html(sec_title("📂", "Filtered Dataset", f"{len(dff):,} records"))
cols_show = [c for c in ["name","brand_clean","year","car_age","price_lakh",
             "km_driven","fuel","seller_type","transmission","owner","city"]
             if c in dff.columns]
with st.expander("View Dataset", expanded=False):
    st.dataframe(dff[cols_show].reset_index(drop=True),
                 use_container_width=True, height=360)
csv = dff[cols_show].to_csv(index=False).encode("utf-8")
st.download_button("⬇️  Download Filtered Dataset (CSV)",
                   csv, "cariq_filtered.csv", "text/csv")
