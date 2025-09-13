import pandas as pd
import matplotlib.pyplot as plt

# Load data
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

# Plot average profiles
for region in agg_df["region"].unique():
    subset = agg_df[agg_df["region"] == region]

    # Temperature
    plt.figure(figsize=(6, 8))
    plt.plot(subset["temp_adjusted"], subset["pressure_mid"], color="red", marker="o")
    plt.gca().invert_yaxis()
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Pressure (dbar)")
    plt.title(f"Average Temperature Profile\n{region}")
    plt.grid(True)
    plt.show()

    # Salinity
    plt.figure(figsize=(6, 8))
    plt.plot(subset["psal_adjusted"], subset["pressure_mid"], color="blue", marker="o")
    plt.gca().invert_yaxis()
    plt.xlabel("Salinity (psu)")
    plt.ylabel("Pressure (dbar)")
    plt.title(f"Average Salinity Profile\n{region}")
    plt.grid(True)
    plt.show()
