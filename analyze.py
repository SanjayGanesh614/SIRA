import dask.dataframe as dd
import numpy as np
import pandas as pd

# --- Step 1: Load dataset ---
def data_transformation(input_path="argo_profiles_binned_1.csv", output_path="argo_profiles_final.csv"):

    ddf = dd.read_csv(input_path)
    # Interpolate salinity & temp

    ddf[["psal_adjusted", "temp_adjusted"]] = ddf.map_partitions(
        lambda pdf: pdf[["psal_adjusted", "temp_adjusted"]].interpolate(method="linear"),
        meta={"psal_adjusted": "f8", "temp_adjusted": "f8"}
    )
    # --- Step 3: Region labeling ---
    def label_region(lat, lon):
        regions = [
            ("Bay of Bengal",            (0, 25, 80, 100)),
            ("Arabian Sea",              (0, 25, 45, 80)),
            ("Equatorial Indian Ocean",  (-15, 15, 45, 85)),
            ("Central Indian Ocean",     (-15, 15, 70, 100)),
            ("Western Indian Ocean",     (-25, 15, 25, 60)),
            ("Southwest Indian Ocean",   (-40, -5, 25, 60)),
            ("Southeast Indian Ocean",   (-40, -5, 85, 135)),
            ("Southern Ocean",           (-90, -40, 20, 150)),
            ("Australian Waters",        (-45, -10, 110, 150)),
            ("Eastern Indian Ocean",     (-15, 15, 85, 120)),
        ]
        for name, (lat_min, lat_max, lon_min, lon_max) in regions:
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                return name
        return "Other"

    ddf["region"] = ddf.map_partitions(
        lambda pdf: pdf.apply(
            lambda row: label_region(row["latitude"], row["longitude"]), axis=1
        ),
        meta=("region", str)
    )

    # --- Step 4: Z-score anomaly detection ---
    # Compute stats for both salinity & temperature
    stats = (
        ddf.groupby(["region", "pressure_bin"])
        .agg({
            "psal_adjusted": ["mean", "std"],
            "temp_adjusted": ["mean", "std"]
        })
        .reset_index()
    )

    # Flatten multi-level columns
    stats.columns = ["region", "pressure_bin",
                    "salinity_mean", "salinity_std",
                    "temp_mean", "temp_std"]

    # Merge stats back
    ddf = ddf.merge(stats, on=["region", "pressure_bin"])

    # Compute z-scores
    ddf["salinity_z"] = (ddf["psal_adjusted"] - ddf["salinity_mean"]) / ddf["salinity_std"]
    ddf["temp_z"]     = (ddf["temp_adjusted"] - ddf["temp_mean"]) / ddf["temp_std"]

    # Flag anomalies
    def flag_anomalies(pdf, col):
        return np.where(
            pdf[col] > 2, "high",
            np.where(pdf[col] < -2, "low", "normal")
        )

    ddf["salinity_flag"] = ddf.map_partitions(
        lambda pdf: flag_anomalies(pdf, "salinity_z"), meta=("salinity_flag", str)
    )
    ddf["temp_flag"] = ddf.map_partitions(
        lambda pdf: flag_anomalies(pdf, "temp_z"), meta=("temp_flag", str)
    )

    # --- Step 5: Save outputs ---
    ddf.to_csv(output_path, index=False, single_file=True)
