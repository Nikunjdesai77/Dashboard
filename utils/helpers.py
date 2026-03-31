"""
utils/helpers.py
Shared helpers: CSS loader, Plotly chart factory, formatters.
"""

import os, re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# ── Brand colour palette ──────────────────────────────────────────────────────
PALETTE   = ["#7C3AED","#06B6D4","#F59E0B","#10B981","#F43F5E",
             "#4F46E5","#EC4899","#8B5CF6","#14B8A6","#F97316",
             "#3B82F6","#A855F7","#22D3EE","#84CC16","#EAB308"]

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font         = dict(family="Inter, sans-serif", color="#94A3B8", size=12),
    margin       = dict(l=10, r=10, t=44, b=10),
    legend       = dict(
        bgcolor      ="rgba(21,33,58,0.8)",
        bordercolor  ="rgba(148,163,184,0.15)",
        borderwidth  = 1,
        font_size    = 11,
    ),
    xaxis = dict(
        gridcolor="rgba(148,163,184,0.07)",
        zerolinecolor="rgba(148,163,184,0.1)",
        tickfont_color="#64748B",
    ),
    yaxis = dict(
        gridcolor="rgba(148,163,184,0.07)",
        zerolinecolor="rgba(148,163,184,0.1)",
        tickfont_color="#64748B",
    ),
    hoverlabel = dict(
        bgcolor="#1C2D4A", bordercolor="#334155",
        font_color="#E2E8F0", font_size=13,
    ),
    title_font = dict(size=14, color="#E2E8F0", family="Inter, sans-serif"),
    title_x    = 0.01,
)

# ── CSS helpers ───────────────────────────────────────────────────────────────
def load_css(path: str = "assets/styles.css"):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            html(f"<style>\n{f.read()}\n</style>")

def html(content: str):
    # Only minify standard HTML strings (not CSS <style> blocks which need newlines)
    if "<style>" not in content:
        content = content.replace("\n", "")
    st.markdown(content, unsafe_allow_html=True)

# ── Formatters ────────────────────────────────────────────────────────────────
def fmt_price(val: float) -> str:
    if val >= 1e7:  return f"₹{val/1e7:.2f} Cr"
    if val >= 1e5:  return f"₹{val/1e5:.2f} L"
    return f"₹{val:,.0f}"

def fmt_num(val: float) -> str:
    if val >= 1e6:  return f"{val/1e6:.1f}M"
    if val >= 1e3:  return f"{val/1e3:.0f}K"
    return str(int(val))

def fmt_km(val: float) -> str:
    return f"{val/1e3:.1f}K km"

# ── KPI card HTML ─────────────────────────────────────────────────────────────
def kpi_card(icon: str, label: str, value: str, sub: str = "",
             color: str = "violet", delay: int = 0) -> str:
    return f"""<div class="kpi-card {color}" style="animation-delay:{delay}ms">
  <div class="kpi-orb"></div>
  <div class="kpi-icon-wrap">{icon}</div>
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  {"<div class='kpi-sub'>"+sub+"</div>" if sub else ""}
</div>"""

def kpi_row(cards: list) -> str:
    inner = "".join(cards)
    return f'<div class="kpi-grid">{inner}</div>'

# ── Section title ─────────────────────────────────────────────────────────────
def sec_title(icon: str, text: str, badge: str = "") -> str:
    b = f'<span class="sec-title-badge">{badge}</span>' if badge else ""
    return f"""<div class="sec-title">
  <span style="font-size:1.1rem">{icon}</span>
  <span class="sec-title-text">{text}</span>
  {b}
</div>"""

# ── Header HTML ───────────────────────────────────────────────────────────────
def page_header(title: str, subtitle: str,
                stats: list[tuple] | None = None) -> str:
    stats_html = ""
    if stats:
        for val, lbl in stats:
            stats_html += f"""<div class="header-stat">
  <div class="header-stat-val">{val}</div>
  <div class="header-stat-lbl">{lbl}</div>
</div>"""
    return f"""<div class="dash-header">
  <div class="header-left">
    <div class="header-title">{title}</div>
    <div class="header-sub">
      <span class="dot-live"></span>
      {subtitle}
      <span class="header-badge">✓ Live Dataset</span>
    </div>
  </div>
  {"<div class='header-right'>"+stats_html+"</div>" if stats_html else ""}
</div>"""

# ── Badge HTML ────────────────────────────────────────────────────────────────
def badge(text: str, color: str = "violet") -> str:
    return f'<span class="badge badge-{color}">{text}</span>'

# ══════════════════════════════════════════════════════════════════════════════
# CHART FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def _apply(fig: go.Figure, title: str = "", height: int = 360) -> go.Figure:
    layout = dict(BASE_LAYOUT)
    layout["height"] = height
    if title:
        layout["title"] = title
    fig.update_layout(**layout)
    return fig


def chart_brand_price(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    grp = (df.groupby("brand_clean")["selling_price"]
             .median().sort_values(ascending=False).head(top_n).reset_index())
    grp.columns = ["Brand", "Median Price"]
    fig = px.bar(
        grp, x="Median Price", y="Brand", orientation="h",
        color="Median Price",
        color_continuous_scale=["#3B82F6","#7C3AED","#EC4899"],
        text=grp["Median Price"].apply(fmt_price),
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    fig.update_yaxes(categoryorder="total ascending")
    return _apply(fig, "🏷️ Median Price by Brand", 400)


def chart_fuel_donut(df: pd.DataFrame) -> go.Figure:
    grp = df["fuel"].value_counts().reset_index()
    grp.columns = ["Fuel", "Count"]
    fig = px.pie(grp, names="Fuel", values="Count", hole=0.6,
                 color_discrete_sequence=PALETTE)
    fig.update_traces(textposition="outside", textinfo="percent+label",
                      marker_line_color="rgba(0,0,0,0)", marker_line_width=2)
    return _apply(fig, "⛽ Fuel Distribution", 360)


def chart_price_trend(df: pd.DataFrame) -> go.Figure:
    grp = (df.groupby("year")["selling_price"]
             .agg(["median","mean","count"]).reset_index().sort_values("year"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grp["year"], y=grp["mean"],
        mode="lines+markers", name="Avg Price",
        line=dict(color="#7C3AED", width=3),
        marker=dict(size=6, color="#7C3AED"),
        fill="tozeroy", fillcolor="rgba(124,58,237,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=grp["year"], y=grp["median"],
        mode="lines", name="Median Price",
        line=dict(color="#06B6D4", width=2, dash="dot"),
    ))
    return _apply(fig, "📈 Price Trend Over Years", 360)


def chart_city_price(df: pd.DataFrame) -> go.Figure:
    grp = (df.groupby("city")["selling_price"]
             .median().sort_values(ascending=False).reset_index())
    grp.columns = ["City", "Median Price"]
    fig = px.bar(grp, x="City", y="Median Price",
                 color="Median Price",
                 color_continuous_scale=["#3B82F6","#7C3AED","#F43F5E"],
                 text=grp["Median Price"].apply(fmt_price))
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    return _apply(fig, "🏙️ City-wise Median Price", 360)


def chart_km_vs_price(df: pd.DataFrame, n: int = 2000) -> go.Figure:
    s = df.sample(min(n, len(df)), random_state=42)
    fig = px.scatter(
        s, x="km_driven", y="selling_price",
        color="fuel", opacity=0.65,
        color_discrete_sequence=PALETTE,
        trendline="lowess",
        labels={"km_driven":"KM Driven","selling_price":"Price (₹)","fuel":"Fuel"},
    )
    return _apply(fig, "🛣️ KM Driven vs Selling Price", 380)


def chart_correlation(df: pd.DataFrame) -> go.Figure:
    num_cols = [c for c in ["selling_price","car_age","km_driven","owner_num"] if c in df.columns]
    corr = df[num_cols].corr().round(2)
    labels = [c.replace("_"," ").title() for c in num_cols]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale="RdBu", zmid=0,
        text=corr.values, texttemplate="%{text:.2f}",
        hovertemplate="%{y} → %{x}: %{z}<extra></extra>",
        colorbar=dict(
            title="Corr", tickfont_color="#94A3B8", title_font_color="#94A3B8",
        ),
    ))
    return _apply(fig, "🔗 Correlation Matrix", 360)


def chart_depreciation(df: pd.DataFrame, brand: str | None = None) -> go.Figure:
    d = df if brand is None else df[df["brand_clean"] == brand]
    grp = (d.groupby("car_age")["selling_price"]
             .agg(["median","q25","q75"] if False else ["median","std"])
             .reset_index().sort_values("car_age"))
    grp.columns = ["car_age","median","std"]
    upper = grp["median"] + grp["std"]
    lower = (grp["median"] - grp["std"]).clip(lower=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.concat([grp["car_age"], grp["car_age"][::-1]]),
        y=pd.concat([upper, lower[::-1]]),
        fill="toself", fillcolor="rgba(124,58,237,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="±1 Std", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=grp["car_age"], y=grp["median"],
        mode="lines+markers", name="Median Price",
        line=dict(color="#06B6D4", width=3),
        marker=dict(size=7, color="#06B6D4"),
    ))
    lbl = brand if brand else "All Brands"
    return _apply(fig, f"📉 Depreciation — {lbl}", 380)


def chart_seller_box(df: pd.DataFrame) -> go.Figure:
    fig = px.box(df, x="seller_type", y="selling_price",
                 color="seller_type", points="outliers",
                 color_discrete_sequence=PALETTE,
                 labels={"seller_type":"Seller","selling_price":"Price (₹)"})
    fig.update_layout(showlegend=False)
    return _apply(fig, "🏪 Price by Seller Type", 360)


def chart_transmission_violin(df: pd.DataFrame) -> go.Figure:
    fig = px.violin(df, x="transmission", y="selling_price",
                    color="transmission", box=True,
                    color_discrete_sequence=["#7C3AED","#06B6D4"],
                    labels={"transmission":"Transmission","selling_price":"Price (₹)"})
    fig.update_layout(showlegend=False)
    return _apply(fig, "⚙️ Price by Transmission", 360)


def chart_feature_importance(feat_imp: pd.Series) -> go.Figure:
    lbl_map = {
        "car_age":         "Car Age",
        "km_driven":       "KM Driven",
        "km_per_year":     "KM / Year",
        "owner_num":       "Owner Count",
        "fuel_enc":        "Fuel Type",
        "seller_type_enc": "Seller Type",
        "transmission_enc":"Transmission",
        "brand_enc":       "Brand",
    }
    fi = feat_imp.rename(index=lbl_map).sort_values()
    df_fi = fi.reset_index()
    df_fi.columns = ["Feature", "Importance"]
    fig = px.bar(df_fi, x="Importance", y="Feature", orientation="h",
                 color="Importance",
                 color_continuous_scale=["#3B82F6","#7C3AED","#EC4899"],
                 labels={"Importance": "Importance Score", "Feature": ""},
                 text=df_fi["Importance"].apply(lambda v: f"{v:.4f}"))
    fig.update_traces(marker_line_width=0, textposition="outside")
    fig.update_coloraxes(showscale=False)
    return _apply(fig, "🎯 Feature Importance", 340)


def chart_model_comparison(results: dict) -> go.Figure:
    names = list(results.keys())
    r2s   = [results[n]["r2"]   for n in names]
    rmses = [results[n]["rmse"] for n in names]
    # Dynamic palette — enough colours for up to 6 models
    r2_colors   = ["#7C3AED","#06B6D4","#F59E0B","#10B981","#F43F5E","#4F46E5"]
    rmse_colors = ["#F43F5E","#10B981","#4F46E5","#F59E0B","#7C3AED","#06B6D4"]
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["R² Score (higher = better)",
                        "RMSE in ₹ (lower = better)"])
    fig.add_trace(go.Bar(
        x=names, y=r2s,
        marker_color=r2_colors[:len(names)],
        text=[f"{v:.4f}" for v in r2s], textposition="outside",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=names, y=rmses,
        marker_color=rmse_colors[:len(names)],
        text=[fmt_price(v) for v in rmses], textposition="outside",
    ), row=1, col=2)
    fig.update_layout(**{**BASE_LAYOUT, "height": 340, "showlegend": False})
    return fig


def chart_brand_volume(df: pd.DataFrame) -> go.Figure:
    grp = df["brand_clean"].value_counts().head(15).reset_index()
    grp.columns = ["Brand","Count"]
    fig = px.bar(grp, x="Brand", y="Count",
                 color="Count",
                 color_continuous_scale=["#3B82F6","#7C3AED"],
                 text="Count")
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    return _apply(fig, "📊 Listing Volume by Brand", 360)


def chart_price_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="selling_price", nbins=60,
                       color_discrete_sequence=["#7C3AED"],
                       marginal="box",
                       labels={"selling_price":"Selling Price (₹)"})
    fig.update_traces(marker_line_width=0)
    return _apply(fig, "📊 Price Distribution", 380)
