# dask_concatfiles_profileid.py
import dask.dataframe as dd
import dask
import xarray as xr
import pandas as pd
import numpy as np
import glob, os
from dask.distributed import Client
import warnings

def process_file(f):
    try:
        ds = xr.open_dataset(f)
        df = ds.to_dataframe().reset_index()
        bins = np.arange(0, 2001, 50)
        df["pressure_bin"] = pd.cut(df["pres_adjusted"], bins)

        agg_df = (
                df.groupby("pressure_bin", observed=True)[["temp_adjusted", "psal_adjusted"]]
                .mean()
                .reset_index()
                .dropna()
            )

        juld = pd.to_datetime(df["juld"].iloc[0])
        agg_df["juld"] = juld
        agg_df["year"] = juld.year
        agg_df["month"] = juld.month
        agg_df["day"] = juld.day
        agg_df["time"] = juld.time()  

        agg_df["latitude"] = df["latitude"].iloc[0]
        agg_df["longitude"] = df["longitude"].iloc[0]

        agg_df["source_file"] = os.path.basename(f)

        return agg_df
    except Exception as e:
        print(f"Error processing {f}: {e}")
        return pd.DataFrame()

def main_process(save_path="argo_profiles_binned_1.csv",nc_data_dir="argo_nc_files/*.nc"):
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        client = Client()  # safe now on Windows

        data_dir = nc_data_dir
        files = glob.glob(data_dir)

        lazy_results = [dask.delayed(process_file)(f) for f in files]
        ddf = dd.from_delayed(lazy_results)

        # out_path = "argo_profiles_binned.parquet"
        # ddf.to_parquet(out_path, engine="pyarrow", write_index=False)
        ddf.to_csv(save_path, index=False, single_file=True)

        print(f"✅ Saved aggregated dataset")
    except Exception as e:
        print(f"Error in main execution: {e}")

# if __name__ == "__main__":
#     main_process()



#TODO:
#make it such that the  data is stored in sql database 
# create a few different graphs
#search datafram to sql
