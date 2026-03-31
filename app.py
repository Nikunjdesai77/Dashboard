"""
app.py — CarIQ: Main Streamlit Entry Point
Multi-page Streamlit app. This file:
  1. Loads & preprocesses data (cached)
  2. Loads/trains ML model via pipeline (cached)
  3. Injects everything into st.session_state
  4. Renders the Home landing page
"""

import streamlit as st
import os
import sys

# ── MUST be first Streamlit call ──────────────────────────────────────────────
st.set_page_config(
    page_title="CarIQ — Used Car Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "**CarIQ v2.1** — Production-grade Used Car Intelligence Dashboard",
    },
)

sys.path.insert(0, os.path.dirname(__file__))

# ── Load CSS ──────────────────────────────────────────────────────────────────
from config import STYLES_CSS, APP_NAME, APP_VERSION, APP_TAGLINE

if os.path.exists(STYLES_CSS):
    with open(STYLES_CSS, encoding="utf-8") as f:
        import textwrap
        st.markdown(textwrap.dedent(f"<style>\n{f.read()}\n</style>"), unsafe_allow_html=True)

# ── Imports (use new unified modules) ────────────────────────────────────────
from utils.preprocessing import load_raw, preprocess
from model.predict import load_artifacts
from model.train import train
from utils.helpers import fmt_price, fmt_num, html
from utils.logger import get_logger

logger = get_logger("app")

# ── Cache data ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data():
    raw = load_raw()
    return preprocess(raw)

# ── Cache model ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model(_df_len: int):
    mdl, enc, scaler, meta = load_artifacts()
    if mdl is None:
        logger.info("No model artifacts found — training...")
        train(verbose=False)
        mdl, enc, scaler, meta = load_artifacts()
    return mdl, enc, scaler, meta

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("⚡ Loading CarIQ Intelligence Engine…"):
    df = get_data()
    mdl, enc, scaler, meta = get_model(len(df))

# Store globally for all pages
st.session_state.update({
    "df":     df,
    "mdl":    mdl,
    "enc":    enc,
    "scaler": scaler,
    "meta":   meta,
})

# ── Sidebar brand + filters ───────────────────────────────────────────────────
with st.sidebar:
    html(
        f"""
        <div class="sidebar-logo">
          <span class="logo-mark">🚗</span>
          <div class="logo-name">{APP_NAME}</div>
          <div class="logo-sub">{APP_TAGLINE} · v{APP_VERSION}</div>
        </div>
        """
    )

    st.markdown("---")
    st.markdown("### 🎛️ Global Filters")

    # Fuel filter
    all_fuels  = sorted(df["fuel"].dropna().unique())
    fuels_sel  = st.multiselect("Fuel Type", all_fuels, default=all_fuels, key="g_fuel")

    # City filter
    if "city" in df.columns:
        all_cities = sorted(df["city"].dropna().unique())
        city_sel   = st.multiselect("City", all_cities, default=all_cities, key="g_city")
    else:
        city_sel = []

    # Year slider
    yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
    yr_range = st.slider("Year Range", yr_min, yr_max, (yr_min, yr_max), key="g_year")

    # Transmission filter
    all_trans = sorted(df["transmission"].dropna().unique())
    trans_sel = st.multiselect("Transmission", all_trans, default=all_trans, key="g_trans")

    # ── Apply filters ──────────────────────────────────────────────────────────
    mask = (
        df["fuel"].isin(fuels_sel) &
        df["year"].between(yr_range[0], yr_range[1]) &
        df["transmission"].isin(trans_sel)
    )
    if city_sel and "city" in df.columns:
        mask = mask & df["city"].isin(city_sel)

    dff = df[mask].copy()
    st.session_state["dff"] = dff

    st.markdown("---")

    # Active model card in sidebar
    if meta:
        best       = meta["best_name"]
        best_res   = meta["results"][best]
        r2         = best_res["r2"]
        rmse       = best_res["rmse"]
        cv_info    = ""
        if "cv_r2_mean" in best_res:
            cv_info = f"""
                <div class="score-item">
                  <span class="score-label">CV R²</span>
                  <span class="score-val" style="color:#A78BFA">{best_res['cv_r2_mean']:.4f}</span>
                </div>"""
        html(
            f"""
            <div class="model-card best" style="font-size:.82rem;">
              <div class="model-name">🏆 {best}</div>
              <div class="model-scores">
                <div class="score-item">
                  <span class="score-label">R² Score</span>
                  <span class="score-val" style="color:#10B981">{r2:.4f}</span>
                </div>
                <div class="score-item">
                  <span class="score-label">RMSE</span>
                  <span class="score-val" style="color:#F59E0B">₹{rmse/1e5:.2f}L</span>
                </div>
                {cv_info}
              </div>
            </div>
            """
        )

    # Model version info
    if meta and meta.get("version"):
        html(f'<div style="text-align:center;font-size:.7rem;color:#475569;margin-top:.3rem;">Model v{meta["version"]}</div>')

    html(
        f'<div class="dash-footer">{APP_NAME} v{APP_VERSION} · Streamlit + XGBoost</div>'
    )

# ── Guard ─────────────────────────────────────────────────────────────────────
if dff.empty:
    st.error("⚠️ No records match the current sidebar filters. Please adjust and try again.")
    st.stop()

# ── Home page content ─────────────────────────────────────────────────────────
from utils.helpers import (
    kpi_card, kpi_row, page_header, sec_title,
    chart_price_trend, chart_fuel_donut, chart_brand_price, fmt_km,
)

html(page_header(
    "CarIQ — Used Car Intelligence",
    "AI-powered pricing · Market analytics · Depreciation insights · CarDekho Dataset",
    stats=[
        (fmt_num(len(dff)),                         "Listings"),
        (fmt_price(dff["selling_price"].median()),   "Median Price"),
        (str(dff["brand_clean"].nunique()),           "Brands"),
        (str(dff["city"].nunique()) if "city" in dff.columns else "—", "Cities"),
    ],
))

html(
    """
    <div style="background:linear-gradient(135deg,rgba(124,58,237,0.1),rgba(6,182,212,0.08));
                border:1px solid rgba(124,58,237,0.2);border-radius:14px;
                padding:1.2rem 1.8rem;margin-bottom:1.5rem;">
      <p style="color:#94A3B8;font-size:.9rem;margin:0;line-height:1.7;">
        👈 <b style="color:#A78BFA">Use the sidebar</b> to navigate between pages:<br>
        📊 <b>Overview</b> — KPIs &amp; data summary &nbsp;|&nbsp;
        📈 <b>Market Analysis</b> — Deep-dive charts &nbsp;|&nbsp;
        🤖 <b>Price Prediction</b> — AI valuation engine &nbsp;|&nbsp;
        💡 <b>Insights</b> — Market intelligence &nbsp;|&nbsp;
        🧪 <b>Model Lab</b> — Automated training &amp; tuning
      </p>
    </div>
    """
)

# KPI rows
avg_price = dff["selling_price"].mean()
med_price = dff["selling_price"].median()
top_brand = dff["brand_clean"].value_counts().idxmax()
avg_km    = dff["km_driven"].mean()
top_fuel  = dff["fuel"].value_counts().idxmax()
pct_first = (dff["owner_num"] == 1).mean() * 100
n_cities  = dff["city"].nunique() if "city" in dff.columns else 0

html(kpi_row([
    kpi_card("📋", "Total Listings",  fmt_num(len(dff)),       f"{n_cities} cities",             "violet", 50),
    kpi_card("💰", "Average Price",   fmt_price(avg_price),    f"Median {fmt_price(med_price)}",  "cyan",   100),
    kpi_card("🏎️", "Top Brand",       top_brand,               "Most listed",                    "amber",  150),
    kpi_card("🛣️", "Avg KM Driven",   fmt_km(avg_km),          "Odometer avg",                   "emerald",200),
]))
html(kpi_row([
    kpi_card("⛽", "Top Fuel",         top_fuel,                "Dominant fuel",                  "cyan",   250),
    kpi_card("👤", "First-Owner %",   f"{pct_first:.1f}%",      "Of all listings",                "violet", 300),
    kpi_card("📅", "Year Range",       f"{int(dff['year'].min())}–{int(dff['year'].max())}",
                                                                "Mfg. span",                      "amber",  350),
    kpi_card("🏙️", "Cities",          str(n_cities),           "In dataset",                     "emerald",400),
]))

col1, col2 = st.columns(2, gap="medium")
with col1:
    html(sec_title("📈", "Price Trend by Year"))
    st.plotly_chart(chart_price_trend(dff), use_container_width=True)
with col2:
    html(sec_title("⛽", "Fuel Distribution"))
    st.plotly_chart(chart_fuel_donut(dff), use_container_width=True)
