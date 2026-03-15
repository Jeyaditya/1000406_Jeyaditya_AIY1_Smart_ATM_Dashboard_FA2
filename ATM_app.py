import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. PAGE CONFIGURATION & LOGO TWEAK
# ==========================================
try:
    img = Image.open("logo.png")
except:
    img = "🏦" 

st.set_page_config(
    page_title="FinTrust ATM Intelligence", 
    page_icon=img, 
    layout="wide"
)
st.sidebar.image(img, use_container_width=True)

# ==========================================
# 2. DATA LOADING & ROBUST PREPROCESSING
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('atm_cash_management_dataset.csv')
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
        numeric_cols = ['Total_Withdrawals', 'Total_Deposits', 'Previous_Day_Cash_Level', 'Holiday_Flag', 'Special_Event_Flag', 'Nearby_Competitor_ATMs']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
        df = df.dropna(subset=['Date', 'Total_Withdrawals'])
        return df
    except FileNotFoundError:
        st.error("Error: 'atm_cash_management_dataset.csv' not found. Please ensure the file is in the same directory.")
        return pd.DataFrame()

df = load_data()
if df.empty:
    st.stop()

# ==========================================
# 3. SIDEBAR FILTERS 
# ==========================================
st.sidebar.header("Filter Data:")

available_locations = df['Location_Type'].dropna().unique().tolist()
available_days = df['Day_of_Week'].dropna().unique().tolist()

selected_locations = st.sidebar.multiselect("Location Type", options=available_locations, default=available_locations)
selected_days = st.sidebar.multiselect("Day of Week", options=available_days, default=available_days)

filtered_df = df[(df['Location_Type'].isin(selected_locations)) & (df['Day_of_Week'].isin(selected_days))]

# ==========================================
# 4. DARK THEME STYLING
# ==========================================
chart_font_color = "#e2e8f0"

st.markdown("""
    <style>
    .stApp {background-color: #0f172a;}
    h1, h2, h3, h4, h5, h6, label, .stMarkdown, p, span {color: #e2e8f0 !important;}
    .insight-box {background-color: #1e293b; color: #f8fafc; border-left: 5px solid #11CAA0; padding: 20px; border-radius: 5px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 15px;}
    .stTabs [data-baseweb="tab-list"] {background-color: #0f172a;}
    .stTabs [data-baseweb="tab"] {color: #e2e8f0 !important;}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 5. MAIN DASHBOARD HEADER
# ==========================================
st.title("FinTrust Bank: ATM Intelligence & Demand Forecasting")
st.markdown("**Data Analyst:** A Jeyaditya (1000406) | **Course:** Data Mining | **Created for:** FA-2")
st.markdown("This interactive planner bridges exploratory analysis, machine learning clustering, and anomaly detection to optimize ATM cash logistics.")

tab1, tab2, tab3 = st.tabs(["EDA & Trends", "ATM Clustering", "Anomaly Detection"])

# ==========================================
# 6. TAB 1: EDA
# ==========================================
with tab1:
    st.header("Exploratory Data Analysis (EDA)")
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig_hist = px.histogram(filtered_df, x="Total_Withdrawals", nbins=30, color="Location_Type",
                                    title="Distribution of Total Withdrawals by Location",
                                    color_discrete_sequence=px.colors.qualitative.Prism)
            fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=chart_font_color))
            st.plotly_chart(fig_hist, use_container_width=True)
            
            holiday_grouped = filtered_df.groupby('Holiday_Flag')['Total_Withdrawals'].mean().reset_index()
            holiday_grouped['Holiday_Label'] = holiday_grouped['Holiday_Flag'].map({0: 'Normal Day', 1: 'Holiday'})
            fig_hol = px.bar(holiday_grouped, x='Holiday_Label', y='Total_Withdrawals', 
                             title="Average Withdrawals: Normal vs Holiday", color='Holiday_Label',
                             color_discrete_sequence=["#001F3F", "#11CAA0"])
            fig_hol.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=chart_font_color))
            st.plotly_chart(fig_hol, use_container_width=True)
            
        with col2:
            if 'Weather_Condition' in filtered_df.columns:
                # 
                fig_box = px.box(filtered_df, x="Weather_Condition", y="Total_Withdrawals", color="Weather_Condition",
                                 title="Impact of Weather on Withdrawals (Outlier Check)",
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=chart_font_color))
                st.plotly_chart(fig_box, use_container_width=True)

            if 'Previous_Day_Cash_Level' in filtered_df.columns:
                fig_scatter = px.scatter(filtered_df, x="Previous_Day_Cash_Level", y="Total_Withdrawals", 
                                         color="Location_Type", title="Previous Day Cash vs. Today's Withdrawals",
                                         color_discrete_sequence=px.colors.qualitative.Prism)
                fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=chart_font_color))
                st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Time-Series Demand Behavior")
        time_series = filtered_df.groupby('Date')['Total_Withdrawals'].sum().reset_index()
        fig_line = px.line(time_series, x='Date', y='Total_Withdrawals', title="Network-Wide Daily Withdrawals Over Time")
        fig_line.update_traces(line_color='#11CAA0', line_width=3)
        fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=chart_font_color))
        st.plotly_chart(fig_line, use_container_width=True)

# ==========================================
# 7. TAB 2: CLUSTERING ANALYSIS
# ==========================================
with tab2:
    st.header("ATM Demand Clustering (K-Means)")
    
    cluster_features = ['Total_Withdrawals', 'Total_Deposits', 'Previous_Day_Cash_Level', 'Nearby_Competitor_ATMs']
    
    if filtered_df.empty or len(filtered_df) < 3:
        st.warning("Not enough data points available for clustering. Try expanding your filters.")
    else:
        X_ml = filtered_df[cluster_features].copy().fillna(0)
        X_ml['Location_Encoded'] = filtered_df['Location_Type'].astype('category').cat.codes
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_ml)
        
        kmeans = KMeans(n_clusters=min(3, len(X_ml)), random_state=42, n_init=10)
        filtered_df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        cluster_means = filtered_df.groupby('Cluster')['Total_Withdrawals'].mean().sort_values()
        label_map = {cluster_means.index[0]: 'Low-Demand'}
        if len(cluster_means) > 1: label_map[cluster_means.index[1]] = 'Steady-Demand'
        if len(cluster_means) > 2: label_map[cluster_means.index[2]] = 'High-Demand'
        
        filtered_df['Cluster_Label'] = filtered_df['Cluster'].map(label_map)
        
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            # 
            fig_cluster = px.scatter_3d(filtered_df, x='Total_Withdrawals', y='Total_Deposits', z='Previous_Day_Cash_Level',
                                        color='Cluster_Label', hover_data=['Location_Type', 'ATM_ID'],
                                        title="3D View of ATM Clusters",
                                        color_discrete_map={'High-Demand': '#ef4444', 'Steady-Demand': '#11CAA0', 'Low-Demand': '#001F3F'})
            fig_cluster.update_traces(marker=dict(size=5, opacity=0.8))
            fig_cluster.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=chart_font_color), margin=dict(l=0, r=0, b=0, t=40))
            st.plotly_chart(fig_cluster, use_container_width=True)
            
        with col_c2:
            st.markdown("### Cluster Profiles")
            st.dataframe(filtered_df.groupby('Cluster_Label')[['Total_Withdrawals', 'Total_Deposits']].mean().astype(int))
            st.markdown("""
            <div class="insight-box">
            <b>Insights:</b><br>
            • <b>High-Demand:</b> Require frequent cash replenishment.<br>
            • <b>Steady-Demand:</b> Highly predictable cash flow.<br>
            • <b>Low-Demand:</b> Can be restocked less frequently to save logistics costs.
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# 8. TAB 3: ANOMALY DETECTION
# ==========================================
with tab3:
    st.header("Anomaly Detection (Holiday & Event Spikes)")
    
    iso_features = ['Total_Withdrawals', 'Holiday_Flag', 'Special_Event_Flag']
    
    if filtered_df.empty or len(filtered_df) < 5:
         st.warning("Not enough data points available for Anomaly Detection. Please adjust your sidebar filters.")
    else:
        X_iso = filtered_df[iso_features].fillna(0)
        

        iso = IsolationForest(contamination=0.03, random_state=42)
        filtered_df['Anomaly_Score'] = iso.fit_predict(X_iso)
        filtered_df['Is_Anomaly'] = filtered_df['Anomaly_Score'].map({1: 'Normal', -1: 'Anomaly'})
        
        fig_anom = px.scatter(filtered_df, x='Date', y='Total_Withdrawals', color='Is_Anomaly',
                              color_discrete_map={'Normal': '#11CAA0', 'Anomaly': '#ef4444'},
                              hover_data=['Location_Type', 'ATM_ID'],
                              title="Withdrawal Timeline with ML-Detected Anomalies")
        fig_anom.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
        fig_anom.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=chart_font_color))
        st.plotly_chart(fig_anom, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <b>Actionable Outcome:</b> The red points signify statistically irregular demand spikes. Operations Managers can preemptively load extra cash into these specific ATMs prior to events.
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Raw Anomaly Data Table")
        anomalies_df = filtered_df[filtered_df['Is_Anomaly'] == 'Anomaly'][['Date', 'ATM_ID', 'Location_Type', 'Holiday_Flag', 'Total_Withdrawals']].sort_values(by='Date')
        st.dataframe(anomalies_df, use_container_width=True)
