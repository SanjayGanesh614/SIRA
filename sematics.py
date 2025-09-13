# import pandas as pd

# # Load CSV
# df = pd.read_csv("argo_profiles_binned_labeled.csv")

# def row_to_sentence(row):
#     pressure = row["pressure_bin"].strip("()[]").split(", ")
#     depth_range = f"{pressure[0]}–{pressure[1]} dbar"
    
#     return (
#         f"On {row['date']} at {row['time']} UTC, in the {row['region']} "
#         f"(latitude {row['latitude']:.2f}°, longitude {row['longitude']:.2f}°), "
#         f"at a depth of {depth_range}, "
#         f"the observed temperature was {row['temp_adjusted']:.2f} °C "
#         f"and salinity was {row['psal_adjusted']:.2f} PSU."
#     )

# # Apply transformation
# df["summary"] = df.apply(row_to_sentence, axis=1)

# # Save if needed
# df.to_csv("argo_semantic_summary.csv", index=False)

# print(df["summary"].head(5))


import pandas as pd

# Load CSV
def run_semantics(input_path="argo_profiles_final.csv", output_path="argo_semantic_summary_1.csv"):
    df = pd.read_csv(input_path)

    def row_to_sentence(row):
        # Extract depth range from pressure_bin string like "(0, 50]"
        try:
            pressure = row["pressure_bin"].strip("()[]").split(", ")
            depth_range = f"{pressure[0]}–{pressure[1]} dbar"
        except Exception:
            depth_range = str(row["pressure_bin"])

        # Build anomaly notes
        sal_note = ""
        if row["salinity_flag"] == "high":
            sal_note = " (above normal)"
        elif row["salinity_flag"] == "low":
            sal_note = " (below normal)"
            
        temp_note = ""
        if row["temp_flag"] == "high":
            temp_note = " (above normal)"
        elif row["temp_flag"] == "low":
            temp_note = " (below normal)"

        return (
            f"On {row['year']}-{row['month']:02d}-{row['day']:02d} "
            f"at {row.get('time', 'unknown')} UTC, "
            f"in the {row['region']} "
            f"(lat {row['latitude']:.2f}°, lon {row['longitude']:.2f}°), "
            f"at a depth of {depth_range}, "
            f"the temperature was {row['temp_adjusted']:.2f} °C{temp_note} "
            f"and the salinity was {row['psal_adjusted']:.2f} PSU{sal_note}."
        )

    # Apply row-to-sentence
    df["summary"] = df.apply(row_to_sentence, axis=1)

    # Save new semantic summary file
    df.to_csv(output_path, index=False)