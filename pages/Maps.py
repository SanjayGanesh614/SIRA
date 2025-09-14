import pandas as pd
import folium
import streamlit as st
from streamlit_folium import st_folium

# Page config
st.set_page_config(page_title="Buoy Map", layout="wide")

st.title("🌊 Buoy Locations Map")

# --- Cache CSV loading ---
@st.cache_data
def load_data():
    df = pd.read_csv("argo_semantic_summary_1.csv")
    # Preprocess once: keep only unique lat-lon-region
    return df[['latitude', 'longitude', 'region']].drop_duplicates()

unique_locations = load_data()

# --- Cache map creation ---
@st.cache_resource
def create_map(locations):
    m = folium.Map(location=[0, 80], zoom_start=3)
    for _, row in locations.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"{row['region']} ({row['latitude']:.2f}, {row['longitude']:.2f})"
        ).add_to(m)
    return m

m = create_map(unique_locations)

# Show map in Streamlit
st_data = st_folium(m, width=800, height=600)

st.success("✅ Map rendered inside Streamlit!")
