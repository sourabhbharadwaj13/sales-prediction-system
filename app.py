import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from model_pipeline import load_and_clean_data, perform_feature_engineering, train_and_evaluate_models
from data_generator import generate_sales_data
from powerbi_export import generate_powerbi_dataset

# PAGE CONFIG
st.set_page_config(page_title="PredicTive | Sales Forecasting", page_icon="⚡", layout="wide")

# ================================
# CUSTOM CSS INJECTION - UI/UX DESIGN
# ================================
def local_css():
    st.markdown("""
    <style>
        /* Base typography & colors */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
            font-weight: 400;
            color: #E2E8F0;
        }

        /* App Background */
        .stApp {
            background-color: #0F172A;
            background-image: radial-gradient(circle at 15% 50%, rgba(56, 189, 248, 0.05), transparent 25%), 
                              radial-gradient(circle at 85% 30%, rgba(168, 85, 247, 0.05), transparent 25%);
        }

        /* Metric Cards */
        [data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
            font-weight: 800 !important;
            color: #38BDF8 !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 1.1rem !important;
            color: #94A3B8 !important;
        }
        
        /* Headers & Titles */
        h1 {
            color: #F8FAFC !important;
            font-weight: 800 !important;
            letter-spacing: -1px;
            margin-bottom: 0.5rem;
        }
        h2, h3 {
            color: #CBD5E1 !important;
            font-weight: 600 !important;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        
        /* DataFrames */
        .dataframe {
            border-radius: 10px !important;
            overflow: hidden !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }

        /* Buttons Focus */
        .stButton>button {
            background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
            color: white !important;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            transition: all 0.2s ease;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.2);
            background: linear-gradient(135deg, #7DD3FC 0%, #A5B4FC 100%);
            border: none;
        }
        
        /* Sidebar Polish */
        [data-testid="stSidebar"] {
            background-color: #1E293B;
            border-right: 1px solid #334155;
        }
        
        /* Info/Success Banners */
        .stAlert {
            border-radius: 10px;
            border: none;
            border-left: 5px solid;
            background-color: #1E293B;
            color: #E2E8F0;
        }
        [data-baseweb="notification"] {
             border-radius: 10px !important;
        }

    </style>
    """, unsafe_allow_html=True)

local_css()

# Graphic/Color Palette mapping for Plotly
CHART_THEME = "plotly_dark"
COLOR_SEQUENCE = ['#38BDF8', '#818CF8', '#C084FC', '#F472B6', '#2DD4BF', '#FBBF24']

# ================================
# DATA & CACHING LAYERS
# ================================
@st.cache_data
def get_data():
    filepath = "sales_data.csv"
    if not os.path.exists(filepath):
        generate_sales_data(filepath)
    return load_and_clean_data(filepath)

@st.cache_resource
def get_model_and_features(_df_clean):
    df_fe = perform_feature_engineering(_df_clean)
    results, best_model_name, best_model, feature_importance, features = train_and_evaluate_models(df_fe)
    return df_fe, results, best_model_name, best_model, feature_importance, features

df_clean = get_data()

# ================================
# SIDEBAR NAVIGATION
# ================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/9941/9941199.png", width=60) # A general icon
st.sidebar.title("PredicTive Engine")
st.sidebar.markdown("<p style='color:#94A3B8; margin-top:-15px;'>Machine Learning Dashboard</p>", unsafe_allow_html=True)
st.sidebar.divider()

page = st.sidebar.radio("Main Modules", [
    "📊 Data Diagnostics", 
    "📈 Exploratory Insights", 
    "🧠 Model Evaluation", 
    "🔮 Forecasting Engine",
    "🎨 Advanced Charts (Seaborn/Matplotlib)"
])

# ================================
# PAGES
# ================================

if page == "📊 Data Diagnostics":
    st.markdown("<h1>Data Diagnostics Workspace</h1>", unsafe_allow_html=True)
    st.info("System has successfully ingested and cleaned the historical retail dataset.", icon="✅")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Live Data Snapshot")
        st.dataframe(df_clean.head(15), use_container_width=True)
        
    with col2:
        st.subheader("Data Blueprint")
        st.write("Total Records:", f"**{len(df_clean):,}**")
        st.write("Unique Stores:", f"**{df_clean['Store_ID'].nunique()}**")
        st.write("Missing Footfall Replaced:", "**Median Imputation**")
        st.markdown("""
        **Time-Series Features Extracted:**
        - `Month`, `Day`, `Year`, `DayOfWeek`
        - `Is_Weekend` (Binary)
        - `Season` (Categorical)
        """)

elif page == "📈 Exploratory Insights":
    st.markdown("<h1>System Insights & EDA</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 1.1rem;'>Macro-level performance trends across all stores and product lines.</p>", unsafe_allow_html=True)
    
    # Top KPI Metrics using columns for cards
    total_sales = df_clean['Sales_Amount'].sum()
    avg_daily_sales = df_clean.groupby('Date')['Sales_Amount'].sum().mean()
    total_promos = df_clean['Has_Promotion'].sum()
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Lifetime Revenue", f"${total_sales:,.0f}")
    k2.metric("Avg Daily Output", f"${avg_daily_sales:,.0f}")
    k3.metric("Promotions Run", f"{total_promos:,}")
    
    st.divider()
    
    # Charts
    daily_sales = df_clean.groupby('Date')['Sales_Amount'].sum().reset_index()
    fig1 = px.line(daily_sales, x='Date', y='Sales_Amount', 
                   title='Overall System Revenue Velocity',
                   template=CHART_THEME, color_discrete_sequence=['#38BDF8'])
    fig1.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig1, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        promo_sales = df_clean.groupby('Has_Promotion')['Sales_Amount'].mean().reset_index()
        promo_sales['Has_Promotion'] = promo_sales['Has_Promotion'].map({0: 'Standard', 1: 'Promoted'})
        fig2 = px.bar(promo_sales, x='Has_Promotion', y='Sales_Amount', 
                      title='Average Revenue: Promoted vs Standard',
                      color='Has_Promotion', color_discrete_sequence=['#94A3B8', '#A855F7'], template=CHART_THEME)
        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)
        
    with c2:
        cat_sales = df_clean.groupby('Product_Category')['Sales_Amount'].sum().reset_index().sort_values(by='Sales_Amount', ascending=False)
        fig3 = px.bar(cat_sales, x='Product_Category', y='Sales_Amount', 
                      title='Macro Category Performance',
                      color='Product_Category', color_discrete_sequence=COLOR_SEQUENCE, template=CHART_THEME)
        fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

elif page == "🧠 Model Evaluation":
    st.markdown("<h1>Algorithmic Engine Analysis</h1>", unsafe_allow_html=True)
    st.markdown("Visualizing the core logic and predictive power of our background models.")
    
    with st.spinner("Compiling algorithmic matrix..."):
        df_fe, results, best_model_name, best_model, feature_importance, features = get_model_and_features(df_clean)
        
    st.success(f"**Champion Model Identified:** {best_model_name} (Winning by R² Score)")
    
    m1, m2 = st.columns([1.5, 2])
    with m1:
        st.subheader("Algorithmic Performance")
        results_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
        
        # Display best stats
        best_stats = results_df[results_df['Model'] == best_model_name].iloc[0]
        st.markdown(f"**{best_model_name} Top Stats:**")
        b1, b2, b3 = st.columns(3)
        b1.metric("Highest R²", f"{best_stats['R2']:.3f}")
        b2.metric("Lowest RMSE", f"{best_stats['RMSE']:.2f}")
        b3.metric("Lowest MAE", f"{best_stats['MAE']:.2f}")
        
        # R2 Comparison Chart
        fig_r2 = px.bar(results_df, x='Model', y='R2', text='R2', 
                        title='R² Score Comparison (Higher is better)', 
                        color='Model', color_discrete_sequence=['#38BDF8', '#C084FC', '#FBBF24'], template=CHART_THEME)
        fig_r2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_r2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False, margin=dict(t=50, b=0))
        st.plotly_chart(fig_r2, use_container_width=True)
        
    with m2:
        if feature_importance is not None:
            st.subheader(f"Decision Weights ({best_model_name})")
            top_features = feature_importance.head(8).sort_values(by="Importance", ascending=True)
            fig = px.bar(top_features, x='Importance', y='Feature', orientation='h', 
                         color='Importance', color_continuous_scale="Purples", template=CHART_THEME)
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)


elif page == "🔮 Forecasting Engine":
    st.markdown("<h1>Predictive Forecasting Lens</h1>", unsafe_allow_html=True)
    st.write("Select your constraints below to fire the ML model and predict forward-looking revenue trajectories.")
    
    with st.container():
        st.markdown("<div style='background-color:#1E293B; padding:1.5rem; border-radius:10px;'>", unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        store_id = f1.selectbox("Filter Store", df_clean['Store_ID'].unique())
        category = f2.selectbox("Filter Category", df_clean['Product_Category'].unique())
        days_to_predict = f3.slider("Projection Horizon (Days)", min_value=7, max_value=30, value=14)
        run_btn = st.button("Initialize Forecast 🚀", use_container_width=True)
        st.markdown("</div><br>", unsafe_allow_html=True)

    if run_btn:
        df_fe, results, best_model_name, best_model, feature_importance, features = get_model_and_features(df_clean)
        latest_data = df_fe[(df_fe['Store_ID'] == store_id) & (df_fe[f'Product_Category_{category}'] == 1) if f'Product_Category_{category}' in df_fe.columns else df_fe['Store_ID'] == store_id]
        
        if latest_data.empty:
            st.error("Insufficient historical bounds to run a forecast for this combination.")
        else:
            with st.spinner("Running neural inference..."):
                current_date = latest_data['Date'].max()
                future_rows = []
                last_row = latest_data.iloc[-1:].copy()
                
                for i in range(1, days_to_predict + 1):
                    next_date = current_date + pd.Timedelta(days=i)
                    pred_row = last_row.copy()
                    
                    # Advance Date variables
                    pred_row['Date'] = next_date
                    pred_row['Month'] = next_date.month
                    pred_row['Day'] = next_date.day
                    pred_row['DayOfWeek'] = next_date.dayofweek
                    pred_row['Year'] = next_date.year
                    pred_row['Is_Weekend'] = 1 if next_date.dayofweek >= 5 else 0
                    
                    pred_val = best_model.predict(pred_row[features])[0]
                    future_rows.append({'Date': next_date, 'Sales_Amount': pred_val, 'Type': 'Forecast'})
                    
                df_future = pd.DataFrame(future_rows)
                
                df_actual = df_clean[(df_clean['Store_ID'] == store_id) & (df_clean['Product_Category'] == category)].sort_values(by='Date')
                df_actual_last30 = df_actual.tail(30).copy()
                df_actual_last30['Type'] = 'Historical Actuals'
                
                plot_df = pd.concat([df_actual_last30[['Date', 'Sales_Amount', 'Type']], df_future[['Date', 'Sales_Amount', 'Type']]])
                
                fig = px.area(plot_df, x='Date', y='Sales_Amount', color='Type', 
                              category_orders={'Type': ['Historical Actuals', 'Forecast']},
                              color_discrete_sequence=['#38BDF8', '#C084FC'],
                              title=f"Forward Revenue Trajectory: Store {store_id} ({category})")
                
                fig.update_layout(template=CHART_THEME, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                # Add transition line
                fig.add_vline(x=current_date.timestamp() * 1000, line_dash="dash", line_color="#E2E8F0")
                
                st.plotly_chart(fig, use_container_width=True)

elif page == "🎨 Advanced Charts (Seaborn/Matplotlib)":
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    st.markdown("<h1>Advanced Statistical Visualizations</h1>", unsafe_allow_html=True)
    st.write("Deeper statistical perspectives using **Matplotlib** and **Seaborn** visualizations instead of Power BI exports.")
    
    # 1. Heatmap for feature correlation
    st.subheader("Feature Correlation Heatmap")
    df_fe, results, best_model_name, best_model, feature_importance, features = get_model_and_features(df_clean)
    
    # Select main numerical columns for correlation
    corr_cols = ['Sales_Amount', 'Customer_Footfall', 'Is_Holiday', 'Has_Promotion'] + [c for c in df_fe.columns if 'Lag' in c or 'Roll' in c]
    corr_data = df_fe[corr_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#0F172A')
    sns.heatmap(corr_data, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, 
                cbar_kws={'label': 'Correlation Coefficient'}, 
                annot_kws={"color": "white"})
    
    # Style styling for dark mode in matplotlib
    ax.tick_params(colors='white')
    for text in ax.texts:
        if abs(float(text.get_text())) < 0.6:
            text.set_color('white')
        else:
            text.set_color('black')
    st.pyplot(fig)
    
    st.divider()
    
    # 2. Boxplot for Sales Distribution by Category
    st.subheader("Sales Distribution by Category (Boxplot)")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    fig2.patch.set_facecolor('#0F172A')
    ax2.set_facecolor('#1E293B')
    
    sns.boxplot(data=df_clean, x='Product_Category', y='Sales_Amount', ax=ax2, palette='husl')
    ax2.set_title('Revenue Spread across Categories', color='white', pad=20)
    ax2.set_xlabel('Product Category', color='white')
    ax2.set_ylabel('Sales Amount ($)', color='white')
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    st.pyplot(fig2)
    
    st.divider()
    
    # 3. Density Plot for Footfall
    st.subheader("Density Distribution: Customer Footfall")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    fig3.patch.set_facecolor('#0F172A')
    ax3.set_facecolor('#1E293B')
    
    sns.kdeplot(data=df_clean, x='Customer_Footfall', hue='Is_Weekend', fill=True, alpha=0.5, ax=ax3, palette=['#38BDF8', '#A855F7'])
    ax3.set_title('Density of Customer Traffic (Weekday vs Weekend)', color='white', pad=20)
    ax3.set_xlabel('Customer Footfall', color='white')
    ax3.set_ylabel('Density', color='white')
    ax3.tick_params(colors='white')
    ax3.spines['bottom'].set_color('white')
    ax3.spines['left'].set_color('white')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Update legend text color
    legend = ax3.get_legend()
    if legend:
        for text in legend.get_texts():
            plt.setp(text, color='white')
        legend.set_title('Is Weekend (1=Yes)')
        plt.setp(legend.get_title(), color='white')
        
    st.pyplot(fig3)
