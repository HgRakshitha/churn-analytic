import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIG ---
st.set_page_config(
    page_title="ChurnGuard | Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN ANALYTICS CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Plus Jakarta Sans', sans-serif !important; }

    /* Fix Sidebar - Remove Scrolling */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        overflow: hidden !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        overflow: hidden !important;
    }

    /* Main Page Background */
    .stApp {
        background-color: #f1f5f9;
        color: #1e293b;
    }

    /* Custom Header Bar */
    .header-bar {
        background-color: #ffffff;
        padding: 1rem 2rem;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* High-Impact Analytics Cards */
    .analytics-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }

    /* Metric Values */
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
    }

    /* Hide Streamlit Clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar Radio Styling */
    [data-testid="stSidebar"] .stRadio label {
        color: #ffffff !important;
        font-weight: 500 !important;
        padding: 10px 15px !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: #1e293b !important;
        color: white !important;
    }
    [data-testid="stSidebar"] .stRadio label[data-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }

    /* Force all sidebar text to white */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div {
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("churn-project/data/cleaned_churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    return df

@st.cache_resource
def load_model():
    return joblib.load("churn-project/models/rf_model.pkl")

df_raw = load_data()
model = load_model()
feats = list(model.feature_names_in_)
df_raw["risk"] = model.predict_proba(df_raw[feats])[:, 1]

# --- SIDEBAR (NON-SCROLLABLE) ---
with st.sidebar:
    st.markdown("<div style='padding: 20px 0; border-bottom: 1px solid #1e293b;'><h2 style='color: white; font-weight: 800; margin:0;'>🛡️ ChurnGuard</h2></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    nav = st.radio("ANALYTICS ENGINE", ["📊 Dashboard", "💰 Sales Hub", "👤 Intelligence", "🧪 Simulator", "📂 Batch Proc", "⚙️ Audit"], label_visibility="collapsed")
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("v4.5 Stable Core")

# --- MAIN PAGE ---
st.markdown("<div class='header-bar'><div><h3 style='margin:0; font-weight:700;'>Intelligence Command</h3><p style='margin:0; color:#64748b; font-size:0.85rem;'>Real-time retention monitoring active.</p></div><div style='background:#f1f5f9; padding:8px 16px; border-radius:8px; font-weight:600; font-size:0.8rem; color:#475569;'>SERVER: US-EAST-01</div></div>", unsafe_allow_html=True)

if nav == "📊 Dashboard":
    # 4-Column KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
        st.metric("TOTAL RECORDS", f"{len(df_raw):,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with k2:
        st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
        st.metric("CRITICAL RISK", f"{(df_raw['risk'] > 0.6).sum():,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with k3:
        st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
        st.metric("AVG RISK %", f"{df_raw['risk'].mean():.1%}")
        st.markdown("</div>", unsafe_allow_html=True)
    with k4:
        st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
        st.metric("RETENTION %", f"{(1 - df_raw['risk'].mean()):.1%}")
        st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-weight:700; margin-bottom:20px;'>Risk Density by Billing</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.kdeplot(data=df_raw, x="MonthlyCharges", hue="Churn_Yes", fill=True, palette="rocket", ax=ax)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-weight:700; margin-bottom:20px;'>Market Segment</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        df_raw["risk"].apply(lambda x: "High" if x > 0.6 else "Medium" if x > 0.3 else "Low").value_counts().plot(kind='pie', autopct='%1.1f%%', colors=["#3b82f6", "#60a5fa", "#93c5fd"], ax=ax)
        ax.set_ylabel('')
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

elif nav == "👤 Intelligence":
    st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
    target = st.selectbox("Search Identity", df_raw.index)
    u = df_raw.loc[target]
    st.markdown("<div style='display:flex; justify-content:space-between; margin-top:20px;'>", unsafe_allow_html=True)
    st.metric("CHURN RISK", f"{u['risk']:.1%}")
    st.metric("TENURE", f"{u['tenure']} MO")
    st.metric("MONTHLY", f"${u['MonthlyCharges']}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif nav == "🧪 Simulator":
    st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        ten = st.slider("Tenure", 0, 72, 36)
        mon = st.slider("Billing", 20, 150, 75)
    with s2:
        con = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        sup = st.radio("Support", ["Yes", "No"], horizontal=True)
    st.markdown("</div>", unsafe_allow_html=True)

    sim = pd.DataFrame(0, index=[0], columns=feats)
    sim["tenure"], sim["MonthlyCharges"] = ten, mon
    sim["TotalCharges"] = ten * mon
    if con == "One year": sim["Contract_One year"] = 1
    elif con == "Two year": sim["Contract_Two year"] = 1
    if sup == "Yes": sim["TechSupport_Yes"] = 1
    
    p = model.predict_proba(sim)[0, 1]
    st.markdown(f"<div class='analytics-card' style='text-align: center; border-left: 10px solid {'#ef4444' if p > 0.5 else '#10b981'};'>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#64748b;'>PREDICTED RISK</p><h1 style='font-size:6rem; margin:0;'>{p:.1%}</h1>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif nav == "📂 Batch Proc":
    st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
    f = st.file_uploader("Upload Data")
    if f:
        df_b = pd.read_csv(f)
        X_b = df_b.reindex(columns=feats, fill_value=0).apply(pd.to_numeric, errors='coerce').fillna(0)
        df_b["Risk"] = model.predict_proba(X_b)[:, 1]
        st.dataframe(df_b.sort_values("Risk", ascending=False), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif nav == "💰 Sales Hub":
    # Sales KPI Row
    s1, s2, s3, s4 = st.columns(4)
    total_rev = df_raw["TotalCharges"].sum()
    mrr = df_raw["MonthlyCharges"].sum()
    arpu = df_raw["MonthlyCharges"].mean()
    risk_rev = df_raw[df_raw["risk"] > 0.6]["MonthlyCharges"].sum()

    with s1:
        st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
        st.metric("LIFETIME REV", f"${total_rev/1e6:.2f}M")
        st.markdown("</div>", unsafe_allow_html=True)
    with s2:
        st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
        st.metric("MONTHLY REV", f"${mrr/1000:.1f}K")
        st.markdown("</div>", unsafe_allow_html=True)
    with s3:
        st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
        st.metric("REV AT RISK", f"${risk_rev/1000:.1f}K", delta=f"-{(risk_rev/mrr):.1%}", delta_color="inverse")
        st.markdown("</div>", unsafe_allow_html=True)
    with s4:
        st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
        st.metric("AVG ARPU", f"${arpu:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Power BI Charts
    p1, p2 = st.columns([1, 1])
    with p1:
        st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-weight:700; margin-bottom:10px;'>Revenue by Contract Type</h4>", unsafe_allow_html=True)
        rev_con = df_raw.groupby("Contract")["MonthlyCharges"].sum().reset_index()
        fig_rev = px.pie(rev_con, values="MonthlyCharges", names="Contract", 
                         hole=0.5, color_discrete_sequence=px.colors.sequential.Tealgrn)
        fig_rev.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300, showlegend=True)
        st.plotly_chart(fig_rev, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with p2:
        st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-weight:700; margin-bottom:10px;'>Billing Distribution (Density)</h4>", unsafe_allow_html=True)
        fig_dens = px.histogram(df_raw, x="MonthlyCharges", color="Churn_Yes", 
                                marginal="box", opacity=0.7, barmode="overlay",
                                color_discrete_map={"No": "#008080", "Yes": "#FF4B4B"})
        fig_dens.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300, showlegend=False)
        st.plotly_chart(fig_dens, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='font-weight:700; margin-bottom:10px;'>Revenue Trend vs Tenure</h4>", unsafe_allow_html=True)
    trend_df = df_raw.groupby("tenure")["MonthlyCharges"].sum().reset_index()
    fig_trend = px.area(trend_df, x="tenure", y="MonthlyCharges", 
                        line_shape="spline", color_discrete_sequence=["#008080"])
    fig_trend.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
    st.plotly_chart(fig_trend, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif nav == "⚙️ Audit":
    st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
    imp = pd.Series(model.feature_importances_, index=feats).sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    imp.plot(kind='barh', color='#3b82f6', ax=ax)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)