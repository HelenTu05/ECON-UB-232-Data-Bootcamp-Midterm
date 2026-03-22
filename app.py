"""
Yelp Restaurant EDA — Interactive Streamlit Dashboard
======================================================
Usage:
    pip install streamlit plotly pandas
    streamlit run app.py

Data:
    Place 'yelp_restaurants_clean.csv' in the same folder as this file.
    Generate it by running the export cell at the bottom of your notebook.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import os
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Yelp Restaurant EDA",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f0f4ff;
        border-left: 4px solid #1e4db7;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .metric-card h2 { margin: 0; color: #1e4db7; font-size: 2rem; }
    .metric-card p  { margin: 0; color: #555; font-size: 0.85rem; }
    .finding-box {
        background: #fffbea;
        border-left: 4px solid #f0a500;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.93rem;
    }
    h1, h2, h3 { color: #1a2d5a; }
    .stSidebar { background: #1a2d5a; }

    /* 侧边栏所有文字变白色 */
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stRadio label { color: white !important; }
    [data-testid="stSidebar"] .stMultiSelect label { color: white !important; }
    [data-testid="stSidebar"] .stSlider label { color: white !important; }
    [data-testid="stSidebar"] p { color: white !important; }
    [data-testid="stSidebar"] h2 { color: white !important; }
    [data-testid="stSidebar"] small { color: #ccddff !important; }
</style>
""", unsafe_allow_html=True)
# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data():
    csv_path = r"C:\Users\PC\OneDrive\Desktop\Data Bootcamp\Midterm_project\yelp_restaurants_clean.csv"
    df = pd.read_csv(csv_path, low_memory=False)
    df["log_review_count"] = np.log1p(df["review_count"])
    df["log_checkin_count"] = np.log1p(df["checkin_count"])
    df["price_label"] = df["price_num"].map({1: "$", 2: "$$", 3: "$$$", 4: "$$$$"})
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "⚠️ **`yelp_restaurants_clean.csv` not found.**\n\n"
        "Run the export cell at the bottom of your notebook first, "
        "then place the CSV in the same folder as `app.py`."
    )
    st.stop()

PRICE_ORDER = ["$", "$$", "$$$", "$$$$"]
NAVY = "#1a2d5a"
GOLD = "#f0a500"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍽️ Yelp EDA")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "📊 Overview",
            "Q1 — Price Tier",
            "Q2 — Review Volume",
            "Q3 — Cuisine Type",
            "Q4 — Geography",
            "Q5 — Operating Hours",
            "Q6 — Amenity Attributes",
            "Q7 — Review Sentiment",
            "📈 Multivariate",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### 🔍 Global Filters")

    # City filter
    all_cities = sorted(df["city"].dropna().unique())
    city_counts = df["city"].value_counts()
    top_city_list = city_counts[city_counts >= 50].index.tolist()
    city_options = ["All Cities"] + sorted(top_city_list)
    sel_cities = st.multiselect("Cities", city_options, default=["All Cities"])

    # Price filter
    sel_prices = st.multiselect(
        "Price Tier", PRICE_ORDER, default=PRICE_ORDER
    )

    # Min review count
    min_reviews = st.slider("Min. Review Count", 1, 500, 5)

    st.markdown("---")
    st.caption(
        f"**Dataset:** Yelp Open Dataset  \n"
        f"**Restaurants:** {len(df):,}  \n"
        f"**Source:** yelp.com/dataset"
    )

# ── Apply global filters ───────────────────────────────────────────────────────
fdf = df.copy()
if "All Cities" not in sel_cities and sel_cities:
    fdf = fdf[fdf["city"].isin(sel_cities)]
if sel_prices:
    fdf = fdf[fdf["price_label"].isin(sel_prices)]
fdf = fdf[fdf["review_count"] >= min_reviews]

n_filtered = len(fdf)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("🍽️ What Factors Influence Yelp Restaurant Star Ratings?")
    
    st.markdown(
        "An exploratory data analysis of **67,533 restaurants** from the Yelp Open Dataset. "
        "Use the sidebar to filter by city, price tier, or review count."
    )

    st.markdown("""
    ---
    ### 🧭 How to Use This Dashboard

    This interactive dashboard accompanies the Yelp Restaurant EDA notebook and allows you to explore the data dynamically — no code required.

    **🔍 Global Filters (sidebar)**
    Use the filters on the left to narrow down the dataset at any time:
    - **Cities** — focus on one or more specific markets
    - **Price Tier** — compare $ vs $$$$ restaurants
    - **Min. Review Count** — exclude low-signal businesses with very few reviews

    All charts and statistics update automatically based on your selection.

    **📌 Page Guide**

    | Page | What You Can Do |
    |------|----------------|
    | 📊 Overview | See the overall star distribution and a summary of all 7 research questions |
    | Q1 — Price Tier | Compare mean ratings across budget, mid-range, upscale, and luxury tiers |
    | Q2 — Review Volume | Explore how review count correlates with star rating |
    | Q3 — Cuisine Type | Adjust the Top N slider to see which cuisines rank highest and lowest |
    | Q4 — Geography | Filter by city and view a live map of restaurant ratings across the US |
    | Q5 — Operating Hours | See how weekly hours relate to star ratings (hint: fewer hours = higher ratings) |
    | Q6 — Amenity Attributes | Compare the rating impact of outdoor seating, delivery, parking, and more |
    | Q7 — Review Sentiment | See how TextBlob sentiment scores align with star ratings |
    | 📈 Multivariate | Explore the correlation matrix, regression coefficients, and a scatter matrix |

    > 💡 **Tip:** Try filtering to a single city (e.g. Philadelphia) and compare how the patterns change versus the full dataset.
    ---
    """)


    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <h2>{n_filtered:,}</h2><p>Restaurants (filtered)</p></div>""",
            unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <h2>{fdf['stars'].mean():.2f}★</h2><p>Mean Star Rating</p></div>""",
            unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <h2>{int(fdf['review_count'].median())}</h2><p>Median Review Count</p></div>""",
            unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <h2>{fdf['city'].nunique()}</h2><p>Unique Cities</p></div>""",
            unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Star Rating Distribution")
        star_counts = fdf["stars"].value_counts().sort_index().reset_index()
        star_counts.columns = ["Stars", "Count"]
        fig = px.bar(
            star_counts, x="Stars", y="Count",
            color="Stars",
            color_continuous_scale="RdYlGn",
            text="Count",
            template="plotly_white",
        )
        fig.update_traces(textposition="outside", texttemplate="%{text:,}")
        fig.update_layout(
            coloraxis_showscale=False,
            xaxis_title="Star Rating",
            yaxis_title="Number of Restaurants",
            showlegend=False,
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Research Questions")
        questions = {
            "Q1": ("Price Tier", "ANOVA", "F=330.79, p<0.001"),
            "Q2": ("Review Volume", "Pearson r", "r=+0.180, p<0.001"),
            "Q3": ("Cuisine Type", "Group means", "1.45★ spread"),
            "Q4": ("Geography", "ANOVA", "F=29.12, p<0.001"),
            "Q5": ("Operating Hours", "Pearson r", "r=−0.443, p<0.001"),
            "Q6": ("Amenity Attributes", "Mann-Whitney U", "p<0.001 all"),
            "Q7": ("Review Sentiment", "TextBlob + r", "Strong alignment"),
        }
        rows = []
        for q, (topic, method, result) in questions.items():
            rows.append({"Q": q, "Topic": topic, "Method": method, "Key Result": result})
        st.dataframe(
            pd.DataFrame(rows).set_index("Q"),
            use_container_width=True,
            height=280,
        )
        st.markdown(
            '<div class="finding-box">Use the <b>sidebar navigation</b> to explore each question interactively.</div>',
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Q1 — Price Tier
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Q1 — Price Tier":
    st.title("Q1 — Does Price Tier Affect Star Ratings?")

    price_df = fdf.dropna(subset=["price_label"])
    price_df = price_df[price_df["price_label"].isin(PRICE_ORDER)]

    if price_df.empty:
        st.warning("No data for selected filters.")
        st.stop()

    price_stats = (
        price_df.groupby("price_label")["stars"]
        .agg(["mean", "median", "std", "count"])
        .reindex([p for p in PRICE_ORDER if p in price_df["price_label"].unique()])
        .reset_index()
    )
    price_stats.columns = ["Price", "Mean ★", "Median ★", "Std", "n"]

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            price_stats, x="Price", y="Mean ★",
            text="Mean ★",
            color="Mean ★",
            color_continuous_scale=[[0, "#4a90d9"], [1, "#1a2d5a"]],
            template="plotly_white",
            labels={"Price": "Price Tier", "Mean ★": "Mean Star Rating"},
        )
        fig.update_traces(
            texttemplate="%{text:.2f}★",
            textposition="outside",
        )
        fig.update_layout(
            title="Mean Star Rating by Price Tier",
            yaxis_range=[3.0, 4.5],
            coloraxis_showscale=False,
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        palette = ["#aec6e8", "#5b9bd5", "#2e75b6", "#1a2d5a"]
        for i, price in enumerate([p for p in PRICE_ORDER if p in price_df["price_label"].unique()]):
            vals = price_df[price_df["price_label"] == price]["stars"].dropna()
            fig2.add_trace(go.Box(
                y=vals, name=price,
                marker_color=palette[i % len(palette)],
                line_color=palette[i % len(palette)],
                boxmean=True,
            ))
        fig2.update_layout(
            title="Star Distribution by Price Tier",
            yaxis_title="Stars",
            template="plotly_white",
            height=420,
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Stats table
    st.subheader("Summary Statistics")
    st.dataframe(price_stats.set_index("Price").style.format(
        {"Mean ★": "{:.3f}", "Median ★": "{:.1f}", "Std": "{:.3f}", "n": "{:,}"}
    ), use_container_width=True)

    # ANOVA
    groups = [
        price_df[price_df["price_label"] == p]["stars"].dropna()
        for p in PRICE_ORDER if p in price_df["price_label"].unique()
    ]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) >= 2:
        f_val, p_val = stats.f_oneway(*groups)
        sig = "✅ Statistically significant" if p_val < 0.05 else "❌ Not significant"
        st.markdown(
            f'<div class="finding-box">🔬 <b>One-way ANOVA:</b> F = {f_val:.2f}, '
            f'p = {p_val:.2e} — {sig}<br>'
            f'Mid-range ($$/$$$) restaurants outperform both budget ($) and luxury ($$$$). '
            f'Luxury restaurants likely suffer from inflated diner expectations.</div>',
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Q2 — Review Volume
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Q2 — Review Volume":
    st.title("Q2 — Does Review Volume Correlate with Star Ratings?")

    rev_df = fdf.dropna(subset=["stars", "review_count"])

    col1, col2 = st.columns(2)

    with col1:
        sample = rev_df.sample(min(6000, len(rev_df)), random_state=42)
        m, b = np.polyfit(sample["log_review_count"], sample["stars"], 1)
        x_range = np.linspace(sample["log_review_count"].min(), sample["log_review_count"].max(), 100)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sample["log_review_count"], y=sample["stars"],
            mode="markers",
            marker=dict(color=NAVY, opacity=0.15, size=5),
            name="Restaurants",
            hovertemplate="log(reviews)=%{x:.2f}<br>Stars=%{y}",
        ))
        fig.add_trace(go.Scatter(
            x=x_range, y=m * x_range + b,
            mode="lines",
            line=dict(color="#e63946", width=2.5),
            name=f"Trend (slope={m:.3f})",
        ))
        fig.update_layout(
            title="log(1 + Review Count) vs. Star Rating",
            xaxis_title="log(1 + Review Count)",
            yaxis_title="Star Rating",
            template="plotly_white",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        bins = [0, 10, 25, 50, 100, 250, 500, 50000]
        labels = ["1–10", "11–25", "26–50", "51–100", "101–250", "251–500", "500+"]
        rev_df = rev_df.copy()
        rev_df["review_bin"] = pd.cut(rev_df["review_count"], bins=bins, labels=labels)
        rev_bin_stats = (
            rev_df.groupby("review_bin", observed=True)["stars"]
            .agg(["mean", "count"])
            .reset_index()
        )
        rev_bin_stats.columns = ["Bucket", "Mean ★", "n"]

        fig2 = px.bar(
            rev_bin_stats, x="Bucket", y="Mean ★",
            text="Mean ★",
            color="Mean ★",
            color_continuous_scale="viridis",
            template="plotly_white",
            hover_data={"n": True},
        )
        fig2.update_traces(texttemplate="%{text:.2f}★", textposition="outside")
        fig2.update_layout(
            title="Mean Stars by Review Count Bucket",
            yaxis_range=[3.0, 4.3],
            coloraxis_showscale=False,
            height=420,
        )
        st.plotly_chart(fig2, use_container_width=True)

    r_val, p_val = stats.pearsonr(rev_df["log_review_count"], rev_df["stars"])
    st.markdown(
        f'<div class="finding-box">🔬 <b>Pearson r = {r_val:+.3f}</b> (p = {p_val:.2e}) — '
        f'weak but highly significant positive correlation.<br>'
        f'Restaurants with 500+ reviews average ~4.00★ vs ~3.45★ for those with 1–10 reviews. '
        f'Likely reflects <i>survivorship bias</i>: popular restaurants accumulate more reviews.</div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Q3 — Cuisine Type
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Q3 — Cuisine Type":
    st.title("Q3 — Which Cuisine Categories Rate Highest & Lowest?")

    cuisine_df = fdf.dropna(subset=["cuisine", "stars"])

    # Controls
    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        top_n = st.slider("Show top N cuisines", 10, 30, 20)
        min_n = st.number_input("Min. restaurants per cuisine", 10, 200, 30)

    cuisine_counts = cuisine_df["cuisine"].value_counts()
    top_cuisines = cuisine_counts[cuisine_counts >= min_n].head(top_n).index.tolist()

    cuisine_stats = (
        cuisine_df[cuisine_df["cuisine"].isin(top_cuisines)]
        .groupby("cuisine")["stars"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=True)
        .reset_index()
    )
    cuisine_stats.columns = ["Cuisine", "Mean ★", "n"]

    overall_mean = fdf["stars"].mean()

    fig = px.bar(
        cuisine_stats, x="Mean ★", y="Cuisine",
        orientation="h",
        color="Mean ★",
        color_continuous_scale="RdYlGn",
        text="Mean ★",
        hover_data={"n": True},
        template="plotly_white",
        height=max(450, top_n * 28),
    )
    fig.update_traces(texttemplate="%{text:.2f}★", textposition="outside")
    fig.add_vline(
        x=overall_mean, line_dash="dash", line_color=NAVY,
        annotation_text=f"Overall mean ({overall_mean:.2f}★)",
        annotation_position="top right",
    )
    fig.update_layout(
        title=f"Mean Star Rating by Cuisine Type (Top {top_n})",
        xaxis_range=[cuisine_stats["Mean ★"].min() - 0.1, cuisine_stats["Mean ★"].max() + 0.25],
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    spread = cuisine_stats["Mean ★"].max() - cuisine_stats["Mean ★"].min()
    best = cuisine_stats.iloc[-1]
    worst = cuisine_stats.iloc[0]
    st.markdown(
        f'<div class="finding-box">🔬 <b>{best["Cuisine"]}</b> leads at {best["Mean ★"]:.2f}★ (n={int(best["n"]):,}); '
        f'<b>{worst["Cuisine"]}</b> is lowest at {worst["Mean ★"]:.2f}★ (n={int(worst["n"]):,}). '
        f'Total spread: ≈{spread:.2f} stars. Artisanal/specialty categories attract enthusiast audiences '
        f'while high-volume fast-casual categories face more diverse and critical reviewers.</div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Q4 — Geography
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Q4 — Geography":
    st.title("Q4 — Does Geographic Location Explain Ratings?")

    geo_df = fdf.dropna(subset=["city", "stars"])
    city_min = st.slider("Min. restaurants per city", 10, 200, 50)

    city_counts = geo_df["city"].value_counts()
    valid_cities = city_counts[city_counts >= city_min].index.tolist()
    top_n_cities = st.slider("Show top N cities", 10, 30, 15)
    valid_cities = valid_cities[:top_n_cities * 2]  # pool to pick from

    city_stats = (
        geo_df[geo_df["city"].isin(valid_cities)]
        .groupby("city")["stars"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=True)
        .tail(top_n_cities)
        .reset_index()
    )
    city_stats.columns = ["City", "Mean ★", "n"]
    overall_mean = fdf["stars"].mean()

    fig = px.bar(
        city_stats, x="Mean ★", y="City",
        orientation="h",
        color="Mean ★",
        color_continuous_scale="RdYlBu",
        text="Mean ★",
        hover_data={"n": True},
        template="plotly_white",
        height=max(400, top_n_cities * 30),
    )
    fig.update_traces(texttemplate="%{text:.2f}★", textposition="outside")
    fig.add_vline(
        x=overall_mean, line_dash="dash", line_color="black",
        annotation_text=f"Overall mean ({overall_mean:.2f}★)",
    )
    fig.update_layout(
        title=f"Mean Star Rating by City (Top {top_n_cities})",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Geo scatter
    st.subheader("Geographic Distribution")
    geo_map = fdf.dropna(subset=["latitude", "longitude", "stars"])
    geo_map = geo_map[
        geo_map["latitude"].between(20, 55) &
        geo_map["longitude"].between(-130, -60)
    ]
    sample_map = geo_map.sample(min(10000, len(geo_map)), random_state=42)
    fig2 = px.scatter_mapbox(
        sample_map, lat="latitude", lon="longitude",
        color="stars", color_continuous_scale="RdYlGn",
        size_max=8, zoom=3, height=420,
        hover_data={"city": True, "stars": True, "review_count": True},
        mapbox_style="carto-positron",
        title="Restaurant Ratings — Geographic Scatter",
        range_color=[1, 5],
    )
    fig2.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig2, use_container_width=True)

    city_groups = [
        geo_df[geo_df["city"] == c]["stars"].dropna()
        for c in valid_cities
    ]
    city_groups = [g for g in city_groups if len(g) > 1]
    if len(city_groups) >= 2:
        f_val, p_val = stats.f_oneway(*city_groups)
        st.markdown(
            f'<div class="finding-box">🔬 <b>One-way ANOVA:</b> F = {f_val:.2f}, p = {p_val:.2e}<br>'
            f'City explains a statistically significant portion of variance in star ratings. '
            f'Local review culture, restaurant mix, and cost-of-living all contribute.</div>',
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Q5 — Operating Hours
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Q5 — Operating Hours":
    st.title("Q5 — Do Longer Operating Hours Predict Better Ratings?")

    hours_df = fdf.dropna(subset=["weekly_hours", "stars"])
    hours_df = hours_df[hours_df["weekly_hours"].between(1, 140)]

    col1, col2 = st.columns(2)

    with col1:
        sample_h = hours_df.sample(min(5000, len(hours_df)), random_state=42)
        m, b = np.polyfit(sample_h["weekly_hours"], sample_h["stars"], 1)
        x_r = np.linspace(sample_h["weekly_hours"].min(), sample_h["weekly_hours"].max(), 100)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sample_h["weekly_hours"], y=sample_h["stars"],
            mode="markers",
            marker=dict(color="#008080", opacity=0.2, size=5),
            name="Restaurants",
            hovertemplate="Hours/week=%{x:.0f}<br>Stars=%{y}",
        ))
        fig.add_trace(go.Scatter(
            x=x_r, y=m * x_r + b,
            mode="lines",
            line=dict(color="#e63946", width=2.5),
            name=f"Trend (slope={m:.4f})",
        ))
        fig.update_layout(
            title="Weekly Hours vs. Star Rating",
            xaxis_title="Total Weekly Operating Hours",
            yaxis_title="Star Rating",
            template="plotly_white",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        hours_df2 = hours_df.copy()
        hours_df2["hours_bin"] = pd.cut(
            hours_df2["weekly_hours"],
            bins=[0, 30, 50, 70, 90, 140],
            labels=["<30h", "30–50h", "50–70h", "70–90h", "90h+"],
        )
        bin_stats = (
            hours_df2.groupby("hours_bin", observed=True)["stars"]
            .agg(["mean", "count"])
            .reset_index()
        )
        bin_stats.columns = ["Bin", "Mean ★", "n"]

        fig2 = px.bar(
            bin_stats, x="Bin", y="Mean ★",
            text="Mean ★",
            color="Mean ★",
            color_continuous_scale="viridis_r",
            hover_data={"n": True},
            template="plotly_white",
        )
        fig2.update_traces(texttemplate="%{text:.2f}★", textposition="outside")
        fig2.update_layout(
            title="Mean Stars by Weekly Hours Bin",
            yaxis_range=[3.0, 4.5],
            coloraxis_showscale=False,
            height=420,
        )
        st.plotly_chart(fig2, use_container_width=True)

    r_val, p_val = stats.pearsonr(hours_df["weekly_hours"], hours_df["stars"])
    st.markdown(
        f'<div class="finding-box">🔬 <b>Pearson r = {r_val:+.3f}</b> (p = {p_val:.2e}) — '
        f'<b>negative</b> correlation: more hours → lower ratings.<br>'
        f'Restaurants open &lt;30h/week average ~4.07★ vs ~3.44★ for 70–90h/week. '
        f'Short-hours venues tend to be specialty/artisanal; long-hours venues are disproportionately '
        f'fast-food chains and 24-hour diners.</div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Q6 — Amenity Attributes
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Q6 — Amenity Attributes":
    st.title("Q6 — Do Amenity Attributes Affect Star Ratings?")

    ATTRS = {
        "has_takeout":     "Takeout",
        "has_delivery":    "Delivery",
        "has_reservation": "Reservations",
        "good_for_kids":   "Good for Kids",
        "outdoor_seating": "Outdoor Seating",
        "has_parking":     "Parking",
    }

    results = []
    for col, label in ATTRS.items():
        if col not in fdf.columns:
            continue
        sub = fdf.dropna(subset=[col, "stars"])
        sub = sub[sub[col].isin([0, 1])]
        yes = sub[sub[col] == 1]["stars"]
        no  = sub[sub[col] == 0]["stars"]
        if len(yes) > 10 and len(no) > 10:
            _, p = stats.mannwhitneyu(yes, no, alternative="two-sided")
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            results.append({
                "Attribute": label,
                "Mean (Has)": round(yes.mean(), 3),
                "Mean (No)":  round(no.mean(), 3),
                "Diff":       round(yes.mean() - no.mean(), 3),
                "n (Has)":    len(yes),
                "n (No)":     len(no),
                "p-value":    p,
                "sig":        sig,
            })

    if not results:
        st.warning("Not enough data for selected filters.")
        st.stop()

    attr_df = pd.DataFrame(results).sort_values("Diff", ascending=True)

    col1, col2 = st.columns(2)

    with col1:
        colors = ["#e63946" if d < 0 else "#1a2d5a" for d in attr_df["Diff"]]
        fig = go.Figure(go.Bar(
            x=attr_df["Diff"],
            y=attr_df["Attribute"],
            orientation="h",
            marker_color=colors,
            text=[f"{d:+.3f}{s}" for d, s in zip(attr_df["Diff"], attr_df["sig"])],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Difference: %{x:+.3f}★<br>"
                "<extra></extra>"
            ),
        ))
        fig.add_vline(x=0, line_color="black", line_width=1)
        fig.update_layout(
            title="Star Rating Difference (Has Attribute vs. Doesn't)",
            xaxis_title="Mean Stars (Yes) − Mean Stars (No)",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        for _, row in attr_df.iterrows():
            fig2.add_trace(go.Bar(
                name="Has Attribute",
                x=[row["Attribute"]],
                y=[row["Mean (Has)"]],
                marker_color=NAVY,
                showlegend=(_ == attr_df.index[0]),
            ))
            fig2.add_trace(go.Bar(
                name="No Attribute",
                x=[row["Attribute"]],
                y=[row["Mean (No)"]],
                marker_color="#f4a261",
                showlegend=(_ == attr_df.index[0]),
            ))
        fig2.update_layout(
            barmode="group",
            title="Mean Stars: Has vs. Doesn't Have",
            yaxis_range=[3.2, 4.1],
            yaxis_title="Mean Stars",
            template="plotly_white",
            height=400,
            xaxis_tickangle=-25,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Full table
    display_df = attr_df[["Attribute", "Mean (Has)", "Mean (No)", "Diff", "n (Has)", "n (No)", "sig"]].set_index("Attribute")
    st.dataframe(
        display_df.style.format({"Mean (Has)": "{:.3f}", "Mean (No)": "{:.3f}", "Diff": "{:+.3f}"}),
        use_container_width=True,
    )

    best_attr = attr_df.iloc[-1]
    worst_attr = attr_df.iloc[0]
    st.markdown(
        f'<div class="finding-box">🔬 <b>{best_attr["Attribute"]}</b> has the largest positive effect '
        f'({best_attr["Diff"]:+.3f}★, {best_attr["sig"]}); '
        f'<b>{worst_attr["Attribute"]}</b> has the largest negative effect '
        f'({worst_attr["Diff"]:+.3f}★, {worst_attr["sig"]}). '
        f'Upscale amenities signal fine-dining; delivery signals fast-food/chain status.</div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Q7 — Sentiment
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Q7 — Review Sentiment":
    st.title("Q7 — Does Review Sentiment Align with Star Ratings?")

    if "mean_sentiment" not in fdf.columns:
        st.info(
            "💡 Sentiment data not found in your CSV. "
            "Make sure to include `mean_sentiment` when exporting from the notebook. "
            "See the export cell instructions below."
        )
        st.markdown("""
        **Add this to your notebook export cell:**
        ```python
        # Merge sentiment into df before exporting
        biz_sentiment = rev_rest.groupby('business_id')['sentiment'].mean().rename('mean_sentiment')
        df = df.merge(biz_sentiment, on='business_id', how='left')
        ```
        """)
        st.stop()

    sent_df = fdf.dropna(subset=["mean_sentiment", "stars"])

    col1, col2 = st.columns(2)

    with col1:
        sent_by_star = (
            sent_df.groupby("stars")["mean_sentiment"]
            .mean()
            .reset_index()
        )
        sent_by_star.columns = ["Stars", "Mean Sentiment"]
        fig = px.bar(
            sent_by_star, x="Stars", y="Mean Sentiment",
            color="Mean Sentiment",
            color_continuous_scale="RdYlGn",
            text="Mean Sentiment",
            template="plotly_white",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(
            title="Mean Sentiment Score by Business Star Rating",
            coloraxis_showscale=False,
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sample_s = sent_df.sample(min(4000, len(sent_df)), random_state=42)
        m, b = np.polyfit(sample_s["mean_sentiment"], sample_s["stars"], 1)
        x_r = np.linspace(sample_s["mean_sentiment"].min(), sample_s["mean_sentiment"].max(), 100)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=sample_s["mean_sentiment"], y=sample_s["stars"],
            mode="markers",
            marker=dict(color="#008080", opacity=0.2, size=5),
            name="Restaurants",
        ))
        fig2.add_trace(go.Scatter(
            x=x_r, y=m * x_r + b,
            mode="lines",
            line=dict(color="#e63946", width=2.5),
            name=f"Trend (slope={m:.3f})",
        ))
        fig2.update_layout(
            title="Business Mean Sentiment vs. Business Star Rating",
            xaxis_title="Mean Review Sentiment (TextBlob Polarity)",
            yaxis_title="Business Stars",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)

    r_val, p_val = stats.pearsonr(sent_df["mean_sentiment"], sent_df["stars"])
    st.markdown(
        f'<div class="finding-box">🔬 <b>Pearson r = {r_val:+.3f}</b> (p = {p_val:.2e})<br>'
        f'Sentiment polarity rises monotonically across star tiers — validating that review text '
        f'and numeric ratings capture consistent signals. TextBlob is a lexical model; '
        f'a transformer-based model would likely show an even stronger correlation.</div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Multivariate
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Multivariate":
    st.title("📈 Multivariate Analysis — Correlation & Regression")

    num_cols_available = [
        c for c in [
            "stars", "log_review_count", "price_num", "weekly_hours",
            "log_checkin_count", "has_takeout", "has_delivery",
            "has_reservation", "outdoor_seating", "has_parking",
        ] if c in fdf.columns
    ]

    corr = fdf[num_cols_available].corr()

    col1, col2 = st.columns(2)

    with col1:
        fig = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".2f",
            aspect="auto",
            template="plotly_white",
        )
        fig.update_layout(
            title="Correlation Matrix",
            height=500,
            coloraxis_colorbar_title="r",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # OLS regression coefficients
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score

        FEATURES = [c for c in num_cols_available if c != "stars"]
        reg_df = fdf[FEATURES + ["stars"]].dropna()
        if len(reg_df) > 50:
            X = reg_df[FEATURES].values
            y = reg_df["stars"].values
            X_std = StandardScaler().fit_transform(X)
            model = LinearRegression().fit(X_std, y)
            r2 = r2_score(y, model.predict(X_std))

            coef_df = (
                pd.DataFrame({"Feature": FEATURES, "Beta": model.coef_})
                .sort_values("Beta")
            )
            colors = ["#e63946" if c < 0 else NAVY for c in coef_df["Beta"]]

            fig2 = go.Figure(go.Bar(
                x=coef_df["Beta"],
                y=coef_df["Feature"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f}" for v in coef_df["Beta"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>β = %{x:+.3f}<extra></extra>",
            ))
            fig2.add_vline(x=0, line_color="black", line_width=1)
            fig2.update_layout(
                title=f"Standardized Regression Coefficients (R² = {r2:.3f})",
                xaxis_title="Standardized β",
                template="plotly_white",
                height=500,
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown(
                f'<div class="finding-box">🔬 <b>OLS R² = {r2:.3f}</b> on {len(reg_df):,} observations.<br>'
                f'Structural features explain ~{r2*100:.0f}% of variance in star ratings. '
                f'<b>Weekly hours</b> is the strongest negative predictor; '
                f'outdoor seating and reservations are the strongest positive predictors.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("Not enough complete cases for regression with current filters.")

    # Pairplot substitute: scatter matrix
    st.subheader("Feature Relationships — Scatter Matrix")
    pair_cols = st.multiselect(
        "Select features",
        num_cols_available,
        default=["stars", "log_review_count", "weekly_hours", "price_num"],
    )
    if len(pair_cols) >= 2:
        sample_p = fdf[pair_cols].dropna().sample(min(3000, len(fdf.dropna(subset=pair_cols))), random_state=42)
        fig3 = px.scatter_matrix(
            sample_p,
            dimensions=pair_cols,
            color=sample_p.get("stars", None) if "stars" in pair_cols else None,
            color_continuous_scale="RdYlGn",
            opacity=0.3,
            template="plotly_white",
            height=600,
        )
        fig3.update_traces(marker_size=3, diagonal_visible=False)
        st.plotly_chart(fig3, use_container_width=True)