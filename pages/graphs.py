import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("🌊 Argo Profiles Dashboard")

# Load CSV directly
df = pd.read_csv("argo_profiles_final.csv")

# Convert pressure_bin into numeric midpoint
def get_midpoint(bin_str):
    bin_str = bin_str.strip("()[]")
    low, high = bin_str.split(",")
    return (float(low) + float(high)) / 2

df["pressure_mid"] = df["pressure_bin"].apply(get_midpoint)

# Group by region and pressure_mid, then compute mean
agg_df = df.groupby(["region", "pressure_mid"]).agg({
    "temp_adjusted": "mean",
    "psal_adjusted": "mean"
}).reset_index()

# Select region to plot
regions = agg_df["region"].unique()
selected_region = st.selectbox("Select Region", regions)

subset = agg_df[agg_df["region"] == selected_region]

# Plot Temperature
st.subheader(f"Average Temperature Profile - {selected_region}")
fig, ax = plt.subplots(figsize=(6, 8))
ax.plot(subset["temp_adjusted"], subset["pressure_mid"], color="red", marker="o")
ax.invert_yaxis()
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Pressure (dbar)")
ax.grid(True)
st.pyplot(fig)

# Plot Salinity
st.subheader(f"Average Salinity Profile - {selected_region}")
fig2, ax2 = plt.subplots(figsize=(6, 8))
ax2.plot(subset["psal_adjusted"], subset["pressure_mid"], color="blue", marker="o")
ax2.invert_yaxis()
ax2.set_xlabel("Salinity (psu)")
ax2.set_ylabel("Pressure (dbar)")
ax2.grid(True)
st.pyplot(fig2)
