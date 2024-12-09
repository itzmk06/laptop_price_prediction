import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import numpy as np

st.set_page_config(page_title="Laptop Dashboard 3.0", layout="wide", page_icon="💻")

@st.cache_data
def load_data():
    df = pd.read_pickle('df.pkl')
    pipe = pickle.load(open('pipe.pkl', 'rb'))
    return df, pipe

df, pipe = load_data()

# Preprocess Data
df['Price'] = df['Price'].replace({',': '', '₹': ''}, regex=True).astype(float)
df['Ram'] = df['Ram'].astype(int)
df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
df.dropna(subset=['Weight'], inplace=True)

# Sidebar for User Input
company = st.sidebar.selectbox("Brand", df['Company'].unique())
type_name = st.sidebar.selectbox("Type", df['TypeName'].unique())
cpu = st.sidebar.selectbox("CPU", df['Cpu'].unique())
ram = st.sidebar.selectbox("RAM (GB)", sorted(df['Ram'].unique()))
gpu = st.sidebar.selectbox("GPU", df['Gpu'].unique())
os = st.sidebar.selectbox("Operating System", df['OpSys'].unique())
weight = st.sidebar.slider("Weight (kg)", float(df['Weight'].min()), float(df['Weight'].max()), 2.5)
touchscreen = st.sidebar.radio("Touchscreen", ['Yes', 'No'])
ips = st.sidebar.radio("IPS Display", ['Yes', 'No'])
ppi = st.sidebar.slider("PPI", int(df['ppi'].min()), int(df['ppi'].max()), 250)

touchscreen = 1 if touchscreen == 'Yes' else 0
ips = 1 if ips == 'Yes' else 0

input_data = np.array([[company, type_name, cpu, ram, gpu, os, weight, touchscreen, ips, ppi]])
predicted_price = np.exp(pipe.predict(input_data)) / 1e5

# Display Predicted Price
st.markdown(f"""
<style>
.predicted-price {{
    font-size: 2.4em;
    font-weight: 600;
    color: #2a9d8f;
    text-align: center;
    margin-top: 20px;
    animation: fadeIn 2s ease-in-out;
}}
@keyframes fadeIn {{
    0% {{ opacity: 0; }}
    100% {{ opacity: 1; }}
}}
</style>
<h1 class="predicted-price">
    Predicted Price: ₹{predicted_price[0]:,.2f} lakhs 💸
</h1>
""", unsafe_allow_html=True)

# Plots
fig_sunburst = px.sunburst(df, path=['Company', 'TypeName', 'Cpu'], values='Price', title="Price Distribution by Company, Type, and CPU")

fig_3dscatter = go.Figure(data=[go.Scatter3d(
    x=df['Weight'], y=df['ppi'], z=df['Price'],
    mode='markers',
    marker=dict(
        size=df['Ram'],
        color=df['Price'],
        colorscale='Viridis',
        opacity=0.8
    ),
    text=df['Company']
)])

fig_boxplot = px.box(df, x="Company", y="Price", color="Company", title="Price Distribution by Company")
fig_scatter = px.scatter(df, x="Weight", y="Price", color="TypeName", size="Ram", title="Weight vs Price")
fig_histogram = px.histogram(df, x="Price", nbins=50, title="Price Distribution")
fig_violin = px.violin(df, x="Cpu", y="Price", box=True, points="all", title="Price vs CPU Type")
fig_pie = px.pie(df, names='TypeName', title="Laptop Types Distribution")
fig_treemap = px.treemap(df, path=['Company', 'TypeName'], values='Price', title="Price Distribution by Company and Type")
fig_bar = px.bar(df, x='Company', y='Price', color='Company', title="Company vs Price")

# Display Plots
st.plotly_chart(fig_sunburst)
st.plotly_chart(fig_3dscatter)
st.plotly_chart(fig_boxplot)
st.plotly_chart(fig_scatter)
st.plotly_chart(fig_histogram)
st.plotly_chart(fig_violin)
st.plotly_chart(fig_pie)
st.plotly_chart(fig_treemap)
st.plotly_chart(fig_bar)
