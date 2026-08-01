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
    # Look next to this file first so the app runs from any working directory,
    # then fall back to a plain relative path.
    here = os.path.dirname(os.path.abspath(__file__))
    for path in [
        os.path.join(here, "data", "yelp_restaurants_clean.csv"),
        os.path.join("data", "yelp_restaurants_clean.csv"),
        os.path.join(here, "yelp_restaurants_clean.csv"),
        "yelp_restaurants_clean.csv",
    ]:
        if os.path.exists(path):
            df = pd.read_csv(path, low_memory=False)
            df["log_review_count"] = np.log1p(df["review_count"])
            df["log_checkin_count"] = np.log1p(df["checkin_count"])
            df["price_label"] = df["price_num"].map({1: "$", 2: "$$", 3: "$$$", 4: "$$$$"})
            return df
    raise FileNotFoundError("yelp_restaurants_clean.csv not found.")

try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "**`data/yelp_restaurants_clean.csv` not found.**\n\n"
        "It ships with the repository. If it is missing, regenerate it by running "
        "the export cell at the end of `notebooks/01_yelp_eda.ipynb`."
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
            "Q8 — Rating Over Time",
            "Q9 — Popularity vs. Quality",
            "📈 Multivariate",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### 🔍 Global Filters")

    all_cities = sorted(df["city"].dropna().unique())
    city_counts = df["city"].value_counts()
    top_city_list = city_counts[city_counts >= 50].index.tolist()
    city_options = ["All Cities"] + sorted(top_city_list)
    sel_cities = st.multiselect("Cities", city_options, default=["All Cities"])

    sel_prices = st.multiselect("Price Tier", PRICE_ORDER, default=PRICE_ORDER)
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

    **🔍 Global Filters (sidebar)** — All charts update automatically based on your selection:
    - **Cities** — focus on one or more specific markets
    - **Price Tier** — compare $ vs $$$$ restaurants
    - **Min. Review Count** — exclude low-signal businesses

    **📌 Page Guide**

    | Page | What You Can Do |
    |------|----------------|
    | 📊 Overview | Overall star distribution + summary of all 9 research questions |
    | Q1 — Price Tier | Compare mean ratings across budget, mid-range, upscale, and luxury tiers |
    | Q2 — Review Volume | Explore how review count correlates with star rating |
    | Q3 — Cuisine Type | Adjust the Top N slider to see which cuisines rank highest and lowest |
    | Q4 — Geography | Filter by city and view a live map of restaurant ratings |
    | Q5 — Operating Hours | See how weekly hours relate to star ratings |
    | Q6 — Amenity Attributes | Compare the rating impact of outdoor seating, delivery, parking, and more |
    | Q7 — Review Sentiment | See how TextBlob sentiment scores align with star ratings |
    | Q8 — Rating Over Time | Explore how ratings evolve over time and the restaurant lifecycle |
    | Q9 — Popularity vs. Quality | Does high check-in traffic equal higher ratings? |
    | 📈 Multivariate | Correlation matrix, regression coefficients, and scatter matrix |

    > 💡 **Tip:** Try filtering to a single city and compare how patterns change vs. the full dataset.
    ---
    """)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card"><h2>{n_filtered:,}</h2><p>Restaurants (filtered)</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><h2>{fdf['stars'].mean():.2f}★</h2><p>Mean Star Rating</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card"><h2>{int(fdf['review_count'].median())}</h2><p>Median Review Count</p></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card"><h2>{fdf['city'].nunique()}</h2><p>Unique Cities</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Star Rating Distribution")
        star_counts = fdf["stars"].value_counts().sort_index().reset_index()
        star_counts.columns = ["Stars", "Count"]
        fig = px.bar(star_counts, x="Stars", y="Count", color="Stars",
                     color_continuous_scale="RdYlGn", text="Count", template="plotly_white")
        fig.update_traces(textposition="outside", texttemplate="%{text:,}")
        fig.update_layout(coloraxis_showscale=False, showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Research Questions")
        questions = {
            "Q1": ("Price Tier",        "ANOVA",            "F=330.79, p<0.001"),
            "Q2": ("Review Volume",     "Pearson r",        "r=+0.180, p<0.001"),
            "Q3": ("Cuisine Type",      "Group means",      "1.45★ spread"),
            "Q4": ("Geography",         "ANOVA",            "F=29.12, p<0.001"),
            "Q5": ("Operating Hours",   "Pearson r",        "r=−0.443, p<0.001"),
            "Q6": ("Amenity Attributes","Mann-Whitney U",   "p<0.001 all"),
            "Q7": ("Review Sentiment",  "TextBlob + r",     "Strong alignment"),
            "Q8": ("Rating Over Time",  "Time-series",      "Honeymoon effect"),
            "Q9": ("Popularity",        "Check-in tiers",   "Popular ≠ overrated"),
        }
        rows = [{"Q": q, "Topic": t, "Method": m, "Key Result": r}
                for q, (t, m, r) in questions.items()]
        st.dataframe(pd.DataFrame(rows).set_index("Q"), use_container_width=True, height=330)
        st.markdown(
            '<div class="finding-box">Use the <b>sidebar navigation</b> to explore each question interactively.</div>',
            unsafe_allow_html=True)

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
        fig = px.bar(price_stats, x="Price", y="Mean ★", text="Mean ★",
                     color="Mean ★", color_continuous_scale=[[0,"#4a90d9"],[1,"#1a2d5a"]],
                     template="plotly_white", labels={"Price":"Price Tier","Mean ★":"Mean Star Rating"})
        fig.update_traces(texttemplate="%{text:.2f}★", textposition="outside")
        fig.update_layout(title="Mean Star Rating by Price Tier", yaxis_range=[3.0,4.5],
                          coloraxis_showscale=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        palette = ["#aec6e8","#5b9bd5","#2e75b6","#1a2d5a"]
        for i, price in enumerate([p for p in PRICE_ORDER if p in price_df["price_label"].unique()]):
            vals = price_df[price_df["price_label"]==price]["stars"].dropna()
            fig2.add_trace(go.Box(y=vals, name=price, marker_color=palette[i%len(palette)],
                                  line_color=palette[i%len(palette)], boxmean=True))
        fig2.update_layout(title="Star Distribution by Price Tier", yaxis_title="Stars",
                           template="plotly_white", height=420, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(price_stats.set_index("Price").style.format(
        {"Mean ★":"{:.3f}","Median ★":"{:.1f}","Std":"{:.3f}","n":"{:,}"}), use_container_width=True)

    groups = [price_df[price_df["price_label"]==p]["stars"].dropna()
              for p in PRICE_ORDER if p in price_df["price_label"].unique()]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) >= 2:
        f_val, p_val = stats.f_oneway(*groups)
        sig = "✅ Statistically significant" if p_val < 0.05 else "❌ Not significant"
        st.markdown(
            f'<div class="finding-box">🔬 <b>One-way ANOVA:</b> F = {f_val:.2f}, p = {p_val:.2e} — {sig}<br>'
            f'Mid-range ($$/$$$) outperform both budget ($) and luxury ($$$$). '
            f'Luxury restaurants likely suffer from inflated diner expectations.</div>',
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Q2 — Review Volume
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Q2 — Review Volume":
    st.title("Q2 — Does Review Volume Correlate with Star Ratings?")

    rev_df = fdf.dropna(subset=["stars","review_count"])
    col1, col2 = st.columns(2)

    with col1:
        sample = rev_df.sample(min(6000,len(rev_df)), random_state=42)
        m, b = np.polyfit(sample["log_review_count"], sample["stars"], 1)
        x_range = np.linspace(sample["log_review_count"].min(), sample["log_review_count"].max(), 100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sample["log_review_count"], y=sample["stars"], mode="markers",
                                 marker=dict(color=NAVY, opacity=0.15, size=5), name="Restaurants"))
        fig.add_trace(go.Scatter(x=x_range, y=m*x_range+b, mode="lines",
                                 line=dict(color="#e63946", width=2.5), name=f"Trend (slope={m:.3f})"))
        fig.update_layout(title="log(1 + Review Count) vs. Star Rating",
                          xaxis_title="log(1 + Review Count)", yaxis_title="Star Rating",
                          template="plotly_white", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        bins   = [0,10,25,50,100,250,500,50000]
        labels = ["1–10","11–25","26–50","51–100","101–250","251–500","500+"]
        rev_df = rev_df.copy()
        rev_df["review_bin"] = pd.cut(rev_df["review_count"], bins=bins, labels=labels)
        rev_bin_stats = (rev_df.groupby("review_bin", observed=True)["stars"]
                         .agg(["mean","count"]).reset_index())
        rev_bin_stats.columns = ["Bucket","Mean ★","n"]
        fig2 = px.bar(rev_bin_stats, x="Bucket", y="Mean ★", text="Mean ★",
                      color="Mean ★", color_continuous_scale="viridis",
                      template="plotly_white", hover_data={"n":True})
        fig2.update_traces(texttemplate="%{text:.2f}★", textposition="outside")
        fig2.update_layout(title="Mean Stars by Review Count Bucket", yaxis_range=[3.0,4.3],
                           coloraxis_showscale=False, height=420)
        st.plotly_chart(fig2, use_container_width=True)

    r_val, p_val = stats.pearsonr(rev_df["log_review_count"], rev_df["stars"])
    st.markdown(
        f'<div class="finding-box">🔬 <b>Pearson r = {r_val:+.3f}</b> (p = {p_val:.2e}) — '
        f'weak but highly significant positive correlation.<br>'
        f'Restaurants with 500+ reviews average ~4.00★ vs ~3.45★ for those with 1–10 reviews. '
        f'Likely reflects <i>survivorship bias</i>.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Q3 — Cuisine Type
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Q3 — Cuisine Type":
    st.title("Q3 — Which Cuisine Categories Rate Highest & Lowest?")

    cuisine_df = fdf.dropna(subset=["cuisine","stars"])
    col_ctrl1, col_ctrl2 = st.columns([1,2])
    with col_ctrl1:
        top_n  = st.slider("Show top N cuisines", 10, 30, 20)
        min_n  = st.number_input("Min. restaurants per cuisine", 10, 200, 30)

    cuisine_counts = cuisine_df["cuisine"].value_counts()
    top_cuisines   = cuisine_counts[cuisine_counts >= min_n].head(top_n).index.tolist()
    cuisine_stats  = (cuisine_df[cuisine_df["cuisine"].isin(top_cuisines)]
                      .groupby("cuisine")["stars"].agg(["mean","count"])
                      .sort_values("mean", ascending=True).reset_index())
    cuisine_stats.columns = ["Cuisine","Mean ★","n"]
    overall_mean = fdf["stars"].mean()

    fig = px.bar(cuisine_stats, x="Mean ★", y="Cuisine", orientation="h",
                 color="Mean ★", color_continuous_scale="RdYlGn", text="Mean ★",
                 hover_data={"n":True}, template="plotly_white",
                 height=max(450, top_n*28))
    fig.update_traces(texttemplate="%{text:.2f}★", textposition="outside")
    fig.add_vline(x=overall_mean, line_dash="dash", line_color=NAVY,
                  annotation_text=f"Overall mean ({overall_mean:.2f}★)", annotation_position="top right")
    fig.update_layout(title=f"Mean Star Rating by Cuisine Type (Top {top_n})",
                      xaxis_range=[cuisine_stats["Mean ★"].min()-0.1, cuisine_stats["Mean ★"].max()+0.25],
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    spread = cuisine_stats["Mean ★"].max() - cuisine_stats["Mean ★"].min()
    best, worst = cuisine_stats.iloc[-1], cuisine_stats.iloc[0]
    st.markdown(
        f'<div class="finding-box">🔬 <b>{best["Cuisine"]}</b> leads at {best["Mean ★"]:.2f}★ (n={int(best["n"]):,}); '
        f'<b>{worst["Cuisine"]}</b> is lowest at {worst["Mean ★"]:.2f}★ (n={int(worst["n"]):,}). '
        f'Total spread: ≈{spread:.2f} stars.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Q4 — Geography
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Q4 — Geography":
    st.title("Q4 — Does Geographic Location Explain Ratings?")

    geo_df     = fdf.dropna(subset=["city","stars"])
    city_min   = st.slider("Min. restaurants per city", 10, 200, 50)
    city_counts = geo_df["city"].value_counts()
    valid_cities = city_counts[city_counts >= city_min].index.tolist()
    top_n_cities = st.slider("Show top N cities", 10, 30, 15)
    valid_cities = valid_cities[:top_n_cities*2]

    city_stats = (geo_df[geo_df["city"].isin(valid_cities)]
                  .groupby("city")["stars"].agg(["mean","count"])
                  .sort_values("mean", ascending=True).tail(top_n_cities).reset_index())
    city_stats.columns = ["City","Mean ★","n"]
    overall_mean = fdf["stars"].mean()

    fig = px.bar(city_stats, x="Mean ★", y="City", orientation="h",
                 color="Mean ★", color_continuous_scale="RdYlBu", text="Mean ★",
                 hover_data={"n":True}, template="plotly_white",
                 height=max(400, top_n_cities*30))
    fig.update_traces(texttemplate="%{text:.2f}★", textposition="outside")
    fig.add_vline(x=overall_mean, line_dash="dash", line_color="black",
                  annotation_text=f"Overall mean ({overall_mean:.2f}★)")
    fig.update_layout(title=f"Mean Star Rating by City (Top {top_n_cities})", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Geographic Distribution")
    geo_map = fdf.dropna(subset=["latitude","longitude","stars"])
    geo_map = geo_map[geo_map["latitude"].between(20,55) & geo_map["longitude"].between(-130,-60)]
    sample_map = geo_map.sample(min(10000,len(geo_map)), random_state=42)
    fig2 = px.scatter_mapbox(sample_map, lat="latitude", lon="longitude",
                              color="stars", color_continuous_scale="RdYlGn",
                              size_max=8, zoom=3, height=420,
                              hover_data={"city":True,"stars":True,"review_count":True},
                              mapbox_style="carto-positron",
                              title="Restaurant Ratings — Geographic Scatter", range_color=[1,5])
    fig2.update_layout(margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig2, use_container_width=True)

    city_groups = [geo_df[geo_df["city"]==c]["stars"].dropna() for c in valid_cities]
    city_groups = [g for g in city_groups if len(g) > 1]
    if len(city_groups) >= 2:
        f_val, p_val = stats.f_oneway(*city_groups)
        st.markdown(
            f'<div class="finding-box">🔬 <b>One-way ANOVA:</b> F = {f_val:.2f}, p = {p_val:.2e}<br>'
            f'City explains a statistically significant portion of variance in star ratings.</div>',
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Q5 — Operating Hours
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Q5 — Operating Hours":
    st.title("Q5 — Do Longer Operating Hours Predict Better Ratings?")

    hours_df = fdf.dropna(subset=["weekly_hours","stars"])
    hours_df = hours_df[hours_df["weekly_hours"].between(1,140)]

    col1, col2 = st.columns(2)
    with col1:
        sample_h = hours_df.sample(min(5000,len(hours_df)), random_state=42)
        m, b = np.polyfit(sample_h["weekly_hours"], sample_h["stars"], 1)
        x_r = np.linspace(sample_h["weekly_hours"].min(), sample_h["weekly_hours"].max(), 100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sample_h["weekly_hours"], y=sample_h["stars"], mode="markers",
                                 marker=dict(color="#008080", opacity=0.2, size=5), name="Restaurants"))
        fig.add_trace(go.Scatter(x=x_r, y=m*x_r+b, mode="lines",
                                 line=dict(color="#e63946", width=2.5), name=f"Trend (slope={m:.4f})"))
        fig.update_layout(title="Weekly Hours vs. Star Rating",
                          xaxis_title="Total Weekly Operating Hours", yaxis_title="Star Rating",
                          template="plotly_white", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        hours_df2 = hours_df.copy()
        hours_df2["hours_bin"] = pd.cut(hours_df2["weekly_hours"],
                                        bins=[0,30,50,70,90,140],
                                        labels=["<30h","30–50h","50–70h","70–90h","90h+"])
        bin_stats = (hours_df2.groupby("hours_bin", observed=True)["stars"]
                     .agg(["mean","count"]).reset_index())
        bin_stats.columns = ["Bin","Mean ★","n"]
        fig2 = px.bar(bin_stats, x="Bin", y="Mean ★", text="Mean ★",
                      color="Mean ★", color_continuous_scale="viridis_r",
                      hover_data={"n":True}, template="plotly_white")
        fig2.update_traces(texttemplate="%{text:.2f}★", textposition="outside")
        fig2.update_layout(title="Mean Stars by Weekly Hours Bin", yaxis_range=[3.0,4.5],
                           coloraxis_showscale=False, height=420)
        st.plotly_chart(fig2, use_container_width=True)

    r_val, p_val = stats.pearsonr(hours_df["weekly_hours"], hours_df["stars"])
    st.markdown(
        f'<div class="finding-box">🔬 <b>Pearson r = {r_val:+.3f}</b> (p = {p_val:.2e}) — '
        f'<b>negative</b> correlation: more hours → lower ratings.<br>'
        f'Restaurants open &lt;30h/week average ~4.07★ vs ~3.44★ for 70–90h/week.</div>',
        unsafe_allow_html=True)

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
        sub = fdf.dropna(subset=[col,"stars"])
        sub = sub[sub[col].isin([0,1])]
        yes = sub[sub[col]==1]["stars"]
        no  = sub[sub[col]==0]["stars"]
        if len(yes) > 10 and len(no) > 10:
            _, p = stats.mannwhitneyu(yes, no, alternative="two-sided")
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            results.append({"Attribute":label, "Mean (Has)":round(yes.mean(),3),
                            "Mean (No)":round(no.mean(),3), "Diff":round(yes.mean()-no.mean(),3),
                            "n (Has)":len(yes), "n (No)":len(no), "p-value":p, "sig":sig})

    if not results:
        st.warning("Not enough data for selected filters.")
        st.stop()

    attr_df = pd.DataFrame(results).sort_values("Diff", ascending=True)
    col1, col2 = st.columns(2)

    with col1:
        colors = ["#e63946" if d < 0 else "#1a2d5a" for d in attr_df["Diff"]]
        fig = go.Figure(go.Bar(x=attr_df["Diff"], y=attr_df["Attribute"], orientation="h",
                               marker_color=colors,
                               text=[f"{d:+.3f}{s}" for d,s in zip(attr_df["Diff"],attr_df["sig"])],
                               textposition="outside"))
        fig.add_vline(x=0, line_color="black", line_width=1)
        fig.update_layout(title="Star Rating Difference (Has Attribute vs. Doesn't)",
                          xaxis_title="Mean Stars (Yes) − Mean Stars (No)",
                          template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        for _, row in attr_df.iterrows():
            fig2.add_trace(go.Bar(name="Has Attribute", x=[row["Attribute"]], y=[row["Mean (Has)"]],
                                  marker_color=NAVY, showlegend=(_==attr_df.index[0])))
            fig2.add_trace(go.Bar(name="No Attribute", x=[row["Attribute"]], y=[row["Mean (No)"]],
                                  marker_color="#f4a261", showlegend=(_==attr_df.index[0])))
        fig2.update_layout(barmode="group", title="Mean Stars: Has vs. Doesn't Have",
                           yaxis_range=[3.2,4.1], yaxis_title="Mean Stars",
                           template="plotly_white", height=400, xaxis_tickangle=-25)
        st.plotly_chart(fig2, use_container_width=True)

    display_df = attr_df[["Attribute","Mean (Has)","Mean (No)","Diff","n (Has)","n (No)","sig"]].set_index("Attribute")
    st.dataframe(display_df.style.format({"Mean (Has)":"{:.3f}","Mean (No)":"{:.3f}","Diff":"{:+.3f}"}),
                 use_container_width=True)
    best_attr, worst_attr = attr_df.iloc[-1], attr_df.iloc[0]
    st.markdown(
        f'<div class="finding-box">🔬 <b>{best_attr["Attribute"]}</b> has the largest positive effect '
        f'({best_attr["Diff"]:+.3f}★); <b>{worst_attr["Attribute"]}</b> has the largest negative effect '
        f'({worst_attr["Diff"]:+.3f}★). Upscale amenities signal fine-dining; delivery signals chain status.</div>',
        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Q7 — Sentiment
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Q7 — Review Sentiment":
    st.title("Q7 — Does Review Sentiment Align with Star Ratings?")

    if "mean_sentiment" not in fdf.columns:
        st.info("💡 Sentiment data not found in your CSV. Add `mean_sentiment` when exporting from the notebook.")
        st.markdown("""
        **Add this to your notebook export cell:**
        ```python
        biz_sentiment = rev_rest.groupby('business_id')['sentiment'].mean().rename('mean_sentiment')
        df = df.merge(biz_sentiment, on='business_id', how='left')
        ```
        """)
        st.stop()

    sent_df = fdf.dropna(subset=["mean_sentiment","stars"])
    col1, col2 = st.columns(2)

    with col1:
        sent_by_star = sent_df.groupby("stars")["mean_sentiment"].mean().reset_index()
        sent_by_star.columns = ["Stars","Mean Sentiment"]
        fig = px.bar(sent_by_star, x="Stars", y="Mean Sentiment",
                     color="Mean Sentiment", color_continuous_scale="RdYlGn",
                     text="Mean Sentiment", template="plotly_white")
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(title="Mean Sentiment Score by Business Star Rating",
                          coloraxis_showscale=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sample_s = sent_df.sample(min(4000,len(sent_df)), random_state=42)
        m, b = np.polyfit(sample_s["mean_sentiment"], sample_s["stars"], 1)
        x_r = np.linspace(sample_s["mean_sentiment"].min(), sample_s["mean_sentiment"].max(), 100)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=sample_s["mean_sentiment"], y=sample_s["stars"], mode="markers",
                                  marker=dict(color="#008080", opacity=0.2, size=5), name="Restaurants"))
        fig2.add_trace(go.Scatter(x=x_r, y=m*x_r+b, mode="lines",
                                  line=dict(color="#e63946", width=2.5), name=f"Trend (slope={m:.3f})"))
        fig2.update_layout(title="Business Mean Sentiment vs. Business Star Rating",
                           xaxis_title="Mean Review Sentiment (TextBlob Polarity)",
                           yaxis_title="Business Stars", template="plotly_white", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    r_val, p_val = stats.pearsonr(sent_df["mean_sentiment"], sent_df["stars"])
    st.markdown(
        f'<div class="finding-box">🔬 <b>Pearson r = {r_val:+.3f}</b> (p = {p_val:.2e})<br>'
        f'Sentiment polarity rises monotonically across star tiers — validating that review text '
        f'and numeric ratings capture consistent signals.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Q8 — Rating Over Time
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Q8 — Rating Over Time":
    st.title("Q8 — Does a Restaurant's Rating Change Over Time?")
    st.markdown("*Note: This page uses the full review-level dataset exported from the notebook. "
                "Make sure `review_year`, `review_month`, and `months_since_open` columns are included in your CSV, "
                "or the static summary below will be shown instead.*")

    has_temporal = all(c in fdf.columns for c in ["review_year", "review_month"])
    has_lifecycle = "months_since_open" in fdf.columns

    # ── Finding 1: Rating trend over time ──
    st.subheader("Finding 1 — Do Ratings Drift Over Time?")
    if has_temporal:
        time_df = fdf.dropna(subset=["review_year","review_month","stars"]).copy()
        time_df["review_year"]  = time_df["review_year"].astype(float).astype(int)
        time_df["review_month"] = time_df["review_month"].astype(float).astype(int)
        monthly = (time_df.groupby(["review_year","review_month"])["stars"]
                   .agg(["mean","count"]).reset_index())
        monthly["date"] = pd.to_datetime(
            monthly["review_year"].astype(str) + "-" + monthly["review_month"].astype(str).str.zfill(2))
        monthly = monthly[monthly["count"] >= 50].sort_values("date")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=monthly["date"], y=monthly["mean"], mode="lines",
                                 line=dict(color="#e63946", width=2), name="Avg Rating"), secondary_y=False)
        fig.add_trace(go.Bar(x=monthly["date"], y=monthly["count"], opacity=0.25,
                             marker_color="#457B9D", name="Review Count"), secondary_y=True)
        fig.update_layout(title="Average Yelp Restaurant Rating Over Time",
                          xaxis_title="Date", template="plotly_white", height=420)
        fig.update_yaxes(title_text="Average Star Rating", range=[3.0,4.5], secondary_y=False)
        fig.update_yaxes(title_text="Number of Reviews", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ `review_year` / `review_month` columns not found. "
                "Export these from the notebook's temporal section.")
        st.markdown(
            '<div class="finding-box">📊 <b>Static finding:</b> Average monthly ratings increased from '
            '~3.75 (2008) to ~4.0 (2019), reflecting platform maturation and gradual quality improvement '
            'as lower-rated businesses close over time.</div>', unsafe_allow_html=True)

    # ── Finding 2: Lifecycle (honeymoon effect) ──
    st.subheader("Finding 2 — The Honeymoon Effect")
    if has_lifecycle:
        lifecycle = (fdf[fdf["months_since_open"].between(0,36)]
                     .groupby("months_since_open")["stars"]
                     .agg(["mean","count","sem"]).reset_index())
        lifecycle = lifecycle[lifecycle["count"] >= 100]
        overall_mean = fdf["stars"].mean()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=lifecycle["months_since_open"],
            y=lifecycle["mean"],
            mode="lines+markers",
            line=dict(color="#2D6A4F", width=2.5),
            error_y=dict(type="data", array=lifecycle["sem"]*1.96, visible=True, color="#2D6A4F"),
            name="Avg Rating",
        ))
        fig2.add_hline(y=overall_mean, line_dash="dash", line_color="gray",
                       annotation_text=f"Overall Mean ({overall_mean:.2f}★)")
        fig2.update_layout(title="Restaurant Rating Trajectory: First 3 Years of Operation",
                           xaxis_title="Months Since First Review", yaxis_title="Average Star Rating",
                           template="plotly_white", height=420, yaxis_range=[3.2,4.2])
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("ℹ️ `months_since_open` column not found.")
        st.markdown(
            '<div class="finding-box">📊 <b>Static finding:</b> In the first 13 months after opening, '
            'ratings run above the overall mean (~3.83★). After month 13, ratings decline and converge '
            'toward the mean — consistent with early reviews from enthusiastic early adopters and personal networks.</div>',
            unsafe_allow_html=True)

    # ── Finding 3: Seasonal variation ──
    st.subheader("Finding 3 — Seasonal Variation")
    if has_temporal:
        seas_df = fdf.dropna(subset=["review_month","stars"]).copy()
        seas_df["review_month"] = seas_df["review_month"].astype(float).astype(int)
        seasonal = seas_df.groupby("review_month")["stars"].mean().reset_index()
        seasonal.columns = ["Month","Mean ★"]
        all_months = pd.DataFrame({"Month": range(1,13)})
        seasonal = all_months.merge(seasonal, on="Month", how="left")
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        seasonal["Month Name"] = seasonal["Month"].apply(lambda x: month_names[x-1])
        annual_mean = seasonal["Mean ★"].mean()
        norm = (seasonal["Mean ★"] - seasonal["Mean ★"].min()) / (seasonal["Mean ★"].max() - seasonal["Mean ★"].min())
        import plotly.colors as pc
        colors = [pc.sample_colorscale("RdBu", [1 - float(n)])[0] for n in norm]
        fig3 = go.Figure(go.Bar(
            x=seasonal["Month Name"], y=seasonal["Mean ★"],
            marker_color=colors,
            text=seasonal["Mean ★"].round(2),
            texttemplate="%{text:.2f}★", textposition="outside",
        ))
        fig3.add_hline(y=annual_mean, line_dash="dash", line_color="black",
                       annotation_text=f"Annual Mean ({annual_mean:.2f}★)",
                       annotation_position="top right")
        fig3.update_layout(
            title="Seasonal Variation in Restaurant Ratings",
            xaxis_title="Month", yaxis_title="Average Star Rating",
            yaxis_range=[3.4, 4.0], template="plotly_white", height=420,
            xaxis=dict(categoryorder="array", categoryarray=month_names),
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.markdown(
            '<div class="finding-box">📊 <b>Static finding:</b> July produces the highest average ratings '
            '(~3.87★) while December is the lowest (~3.78★). The effect is under 0.1 stars — '
            'real but not practically significant.</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="finding-box">🔬 <b>Q8 Conclusion:</b> Yelp ratings are not static. '
        'New restaurants benefit from an early enthusiasm bias that fades within year one. '
        'Platform-wide ratings have crept upward over time. Seasonal effects exist but are minor. '
        '<b>A restaurant\'s rating should always be read alongside its age and review history.</b></div>',
        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Q9 — Popularity vs. Quality
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Q9 — Popularity vs. Quality":
    st.title("Q9 — Does Popularity Equal Quality?")

    # ── Finding 1: Review count vs rating ──
    st.subheader("Finding 1 — Review Count vs. Star Rating")
    col1, col2 = st.columns(2)

    with col1:
        sample = fdf.dropna(subset=["review_count","stars"]).sample(min(6000,len(fdf)), random_state=42)
        m, b = np.polyfit(sample["log_review_count"], sample["stars"], 1)
        x_range = np.linspace(sample["log_review_count"].min(), sample["log_review_count"].max(), 100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sample["log_review_count"], y=sample["stars"], mode="markers",
                                 marker=dict(color="#E76F51", opacity=0.15, size=5), name="Restaurants"))
        fig.add_trace(go.Scatter(x=x_range, y=m*x_range+b, mode="lines",
                                 line=dict(color="black", width=2),
                                 name=f"r={stats.pearsonr(sample['log_review_count'],sample['stars'])[0]:.3f}"))
        fig.update_layout(title="Review Count vs. Star Rating (log scale)",
                          xaxis_title="Log(Review Count)", yaxis_title="Star Rating",
                          template="plotly_white", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        pop_df = fdf.copy()
        pop_df["review_bucket"] = pd.cut(pop_df["review_count"],
                                         bins=[0,10,50,200,500,99999],
                                         labels=["1–10","11–50","51–200","201–500","500+"])
        fig2 = px.box(pop_df.dropna(subset=["review_bucket"]), x="review_bucket", y="stars",
                      color="review_bucket", template="plotly_white",
                      labels={"review_bucket":"Number of Reviews","stars":"Star Rating"},
                      title="Rating Stability by Review Volume")
        fig2.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig2, use_container_width=True)

    r_val, p_val = stats.pearsonr(fdf["log_review_count"].dropna(),
                                   fdf.loc[fdf["log_review_count"].notna(),"stars"])
    st.markdown(
        f'<div class="finding-box">🔬 <b>Pearson r = {r_val:+.3f}</b> (p = {p_val:.2e}) — '
        f'weak positive correlation. Restaurants with 500+ reviews average ~4.25★ with a tight IQR '
        f'vs ~3.45★ and wide variance for 1–10 reviews.</div>', unsafe_allow_html=True)

    # ── Finding 2: Check-in tiers ──
    st.subheader("Finding 2 — Do 'Viral' Restaurants Rate Higher or Lower?")
    checkin_df = fdf[fdf["checkin_count"] > 0].copy()
    if len(checkin_df) > 100:
        checkin_df["popularity_tier"] = pd.qcut(
            checkin_df["checkin_count"], q=4,
            labels=["Low\n(bottom 25%)","Medium-Low","Medium-High","High\n(top 25%)"])

        col3, col4 = st.columns(2)
        with col3:
            fig3 = px.violin(checkin_df, x="popularity_tier", y="stars",
                             color="popularity_tier", box=True,
                             color_discrete_sequence=["#A8DADC","#457B9D","#1D3557","#E63946"],
                             labels={"popularity_tier":"Check-in Popularity Tier","stars":"Star Rating"},
                             title="Rating Distribution by Check-in Popularity",
                             template="plotly_white")
            fig3.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            tier_stats = checkin_df.groupby("popularity_tier")["stars"].agg(["mean","sem"]).reset_index()
            tier_stats.columns = ["Tier","Mean ★","SEM"]
            fig4 = go.Figure(go.Bar(
                x=tier_stats["Tier"], y=tier_stats["Mean ★"],
                error_y=dict(type="data", array=tier_stats["SEM"]*1.96, visible=True),
                marker_color=["#A8DADC","#457B9D","#1D3557","#E63946"],
                text=tier_stats["Mean ★"].round(3),
                texttemplate="%{text:.2f}★", textposition="outside",
            ))
            fig4.update_layout(title="Mean Rating by Check-in Popularity Tier (±95% CI)",
                               yaxis_range=[3.0,4.2], yaxis_title="Mean Star Rating (±95% CI)",
                               xaxis_title="Check-in Popularity Tier",
                               template="plotly_white", height=400)
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown(
            '<div class="finding-box">🔬 The top check-in quartile shows the <b>highest</b> mean rating — '
            'contrary to the "crowded = overrated" hypothesis. Sustained popularity likely reflects '
            'earned reputation. The medium-low tier rates lowest, possibly capturing one-time curiosity visits.</div>',
            unsafe_allow_html=True)
    else:
        st.warning("Not enough check-in data for selected filters.")

    # ── Finding 3: Rating variance ──
    st.subheader("Finding 3 — Rating Variance Decreases with Review Count (Law of Large Numbers)")
    pop_df2 = fdf.copy()
    pop_df2["review_bucket"] = pd.cut(pop_df2["review_count"],
                                       bins=[0,10,50,200,500,99999],
                                       labels=["1–10","11–50","51–200","201–500","500+"])
    variance_df = (pop_df2.dropna(subset=["review_bucket"])
                   .groupby("review_bucket", observed=True)["stars"].std().reset_index())
    variance_df.columns = ["Review Count Bucket","Std Dev"]

    fig5 = px.line(variance_df, x="Review Count Bucket", y="Std Dev", markers=True,
                   labels={"Review Count Bucket":"Number of Reviews","Std Dev":"Standard Deviation of Rating"},
                   title="Rating Variance Decreases as Review Count Grows",
                   template="plotly_white")
    fig5.update_traces(line=dict(color="#E63946", width=2.5), marker=dict(size=10))
    fig5.update_layout(height=380)
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown(
        '<div class="finding-box">🔬 <b>Q9 Conclusion:</b> Popularity and quality are neither equivalent '
        'nor opposites. The most important effect of high review volume is on <b>rating reliability</b>. '
        'The "crowded means overrated" assumption is not supported — sustained traffic reflects and '
        'reinforces genuine quality. <b>High engagement stabilizes and modestly elevates ratings over time.</b></div>',
        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Multivariate
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Multivariate":
    st.title("📈 Multivariate Analysis — Correlation & Regression")

    num_cols_available = [c for c in [
        "stars","log_review_count","price_num","weekly_hours",
        "log_checkin_count","has_takeout","has_delivery",
        "has_reservation","outdoor_seating","has_parking",
    ] if c in fdf.columns]

    corr = fdf[num_cols_available].corr()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        text_auto=".2f", aspect="auto", template="plotly_white")
        fig.update_layout(title="Correlation Matrix", height=500, coloraxis_colorbar_title="r")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
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

            coef_df = pd.DataFrame({"Feature":FEATURES,"Beta":model.coef_}).sort_values("Beta")
            colors = ["#e63946" if c < 0 else NAVY for c in coef_df["Beta"]]

            fig2 = go.Figure(go.Bar(x=coef_df["Beta"], y=coef_df["Feature"], orientation="h",
                                    marker_color=colors,
                                    text=[f"{v:+.3f}" for v in coef_df["Beta"]],
                                    textposition="outside"))
            fig2.add_vline(x=0, line_color="black", line_width=1)
            fig2.update_layout(title=f"Standardized Regression Coefficients (R² = {r2:.3f})",
                               xaxis_title="Standardized β", template="plotly_white", height=500)
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown(
                f'<div class="finding-box">🔬 <b>OLS R² = {r2:.3f}</b> on {len(reg_df):,} observations.<br>'
                f'Structural features explain ~{r2*100:.0f}% of variance in star ratings. '
                f'<b>Weekly hours</b> is the strongest negative predictor; '
                f'outdoor seating and reservations are the strongest positive predictors.</div>',
                unsafe_allow_html=True)
        else:
            st.warning("Not enough complete cases for regression with current filters.")

    st.subheader("Feature Relationships — Scatter Matrix")
    pair_cols = st.multiselect("Select features", num_cols_available,
                               default=["stars","log_review_count","weekly_hours","price_num"])
    if len(pair_cols) >= 2:
        sample_p = fdf[pair_cols].dropna().sample(min(3000,len(fdf.dropna(subset=pair_cols))), random_state=42)
        fig3 = px.scatter_matrix(sample_p, dimensions=pair_cols,
                                 color=sample_p["stars"] if "stars" in pair_cols else None,
                                 color_continuous_scale="RdYlGn", opacity=0.3,
                                 template="plotly_white", height=600)
        fig3.update_traces(marker_size=3, diagonal_visible=False)
        st.plotly_chart(fig3, use_container_width=True)