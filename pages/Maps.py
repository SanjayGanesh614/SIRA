import pandas as pd
import folium
import streamlit as st
from streamlit_folium import st_folium

# Page config
st.set_page_config(page_title="Buoy Map", layout="wide")

st.title("🌊 Buoy Locations Map")

# Load CSV
df = pd.read_csv("argo_semantic_summary_1.csv")

# Keep only unique lat-lon combinations with region
unique_locations = df[['latitude', 'longitude', 'region']].drop_duplicates()

# Create Folium map centered roughly in the Indian Ocean
m = folium.Map(location=[0, 80], zoom_start=3)

# Add markers
for _, row in unique_locations.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"{row['region']} ({row['latitude']:.2f}, {row['longitude']:.2f})"
    ).add_to(m)

# Show map in Streamlit
st_data = st_folium(m, width=800, height=600)

st.success("✅ Map rendered inside Streamlit!")
