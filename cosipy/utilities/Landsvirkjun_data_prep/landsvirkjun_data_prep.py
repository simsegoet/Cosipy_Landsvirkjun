import pandas as pd
import glob
import os
import numpy as np
from cosipy.utilities.config_utils import UtilitiesConfig


UtilitiesConfig.load()
input_folder = UtilitiesConfig.landsvirkjun_data_prep.paths["input_path"]
output_folder = UtilitiesConfig.landsvirkjun_data_prep.paths["output_path"]
output_file_name = UtilitiesConfig.landsvirkjun_data_prep.paths["output_filename"]
safe_individual_csv = UtilitiesConfig.landsvirkjun_data_prep.paths.get("safe_individual_csv", False)




def replace_nans(df):
    """
    Replace nans with mean in the COSIPY relevant columns. And print info.

    Parameters
    ----------
    df: dataframe
    
    Returns
    ------- 
    """
    columns_to_check = ['ps', 't', 'rh', 'sw_in', 'TotalPrecipmweq', 'f', 'lw_in', 'Snowfallm']
    for col in columns_to_check:
        if col in df.columns:
            if df[col].isna().any():
                nan_rows = df[df[col].isna()]
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
                print(f"Column '{col}' contained {len(nan_rows)} NaN values. Replaced by mean. Timestamps of NaNs:")
                print(nan_rows["TIMESTAMP"].dropna().to_list())
        else:
            print(f"Column '{col}' not found in dataset")

def standardize_units_and_save_to_csv(input_csv, output_folder, safe_individual_csv=False):
    """
    Converts units to COSIPY-compatible units, selects and filters Snowheight.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV file.
    output_folder : str
        Directory where outputs are saved.
    safe_individual_csv : bool, keeyword (default: False)
        Decision if individual input files should be safed individually.

    Returns
    -------
    pandas.DataFrame
        The processed DataFrame with converted units and filtered HS column.
    """
        
    try:
        df = pd.read_csv(input_csv)

        if "Time" in df.columns:
            df["TIMESTAMP"] = pd.to_datetime(df["Time"], errors="coerce")
            df = df.drop(columns=["Time"])

        df = df.dropna(axis=1, how="all")    # drop columns that are all NaN

        if "t2" in df.columns and np.nanmean(df["t2"]) < 150:
            df["t2"] = df["t2"] + 273.15
        if "Ts" in df.columns and np.nanmean(df["Ts"]) < 150:
            df["Ts"] = df["Ts"] + 273.15
        if "t" in df.columns and np.nanmean(df["t"]) < 150:
            df["t"] = df["t"] + 273.15

        if "ps" in df.columns and np.nanmean(df["t"]) > 1500:   # make sure ps is in hPa
            df["ps"] = df["ps"] / 100

        # Precip conversions
        if "TotalPrecipmweq" in df.columns:
            mean_val = df["TotalPrecipmweq"].mean()
            df.loc[df["TotalPrecipmweq"] > 0.1, "TotalPrecipmweq"] = mean_val  # replace unphysically high values with mean
            df["TotalPrecipmm"] = df["TotalPrecipmweq"] * 1000.0     # conversion from mweq to mm


        if "Snowfallmweq" in df.columns:
            mean_val = df["Snowfallmweq"].mean()
            df.loc[df["Snowfallmweq"] > 0.1, "Snowfallmweq"] = mean_val    # replace unphysically high values with mean
            density_fresh_snow = np.maximum(109.0 + 6.0 * (df["t"] - 273.16) + 26.0 * np.sqrt(df["f"]), 50.0)   # conversion from mweq to m (Vionnet et al. 2012) 
            df["Snowfallm"] = df["Snowfallmweq"] * 1000 / density_fresh_snow

        # selection of HS and filtering
        if "HS_mod" in df.columns:
            df["HS_sel"] = df["HS_mod"]
        elif "HS" in df.columns:
            df["HS_sel"] = df["HS"]
        else:
            df["HS_sel"] = np.nan

        max_hs = df["HS_sel"].abs().max()
        if max_hs > 25:  # likely cm -> convert to m
            df["HS_sel"] = df["HS_sel"] / 100.0
        df.loc[df["HS_sel"] > 10, "HS_sel"] = np.nan  # remove unphysically high values
        df.loc[df["HS_sel"] == 0, "HS_sel"] = np.nan
        df["HS_sel"] = -df["HS_sel"]  # convention: negative height

        df = apply_hampel_to_hs(df, col="HS_sel", window=15, k=2.0)

        
        for col in ["sw_in", "sw_out", "lw_in", "lw_out"]:
            if col not in df.columns:
                df[col] = None
                
        replace_nans(df)
        
        # bring columns in snesible order
        priority = ["TIMESTAMP", "ps", "t", "rh", "sw_in", "TotalPrecipmm", "f", "lw_in", "Snowfallm", "Albedo_acc", "HS_sel"]
        priority_present = [c for c in priority if c in df.columns]
        rest = [c for c in df.columns if c not in priority_present]
        df = df[priority_present + rest]
                
        # option of saving each input file (year) as a individual .csv file 
        if safe_individual_csv:
            individual_csv_name = os.path.basename(input_csv).replace(".csv", "_rdy_conv.csv")
            individual_csv_path = os.path.join(output_folder, individual_csv_name)
            df.to_csv(individual_csv_path, index=False)
            print(f"Processed {input_csv} to {individual_csv_path}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

    return df


def hampel_filter_series(s: pd.Series, window: int = 15, k: float = 2.0):
    """
    Replace spikes in the time series using a Hampel filter.

    Parameters
    ----------
    s : pandas.Series
        Time series to filter.
    window : int, default 15
        Rolling window length each point is compared to the median of its local window.
    k : float, default 2.0
        Outlier threshold

    Returns
    -------
    filtered : pandas.Series
        Copy of `s` where flagged spikes are replaced by the rolling median.
    spike_mask : pandas.Series of bool
        Boolean mask indicating which points were replaced.
    """
    
    s = s.astype(float)
    med = s.rolling(window=window, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(window=window, center=True, min_periods=1).median()
    scale = 1.4826 * mad      # 1.4826 makes MAD a consistent estimator of std for Gaussian data

    # Avoid zero scale being overly aggressive or dividing by zero
    # Use a small global fallback based on data if scale is zero
    global_scale = np.nanmedian(scale[scale > 0])
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = np.nanstd(s) if np.nanstd(s) > 0 else 1.0
    scale = scale.mask((scale <= 0) | (~np.isfinite(scale)), other=global_scale)

    diff = (s - med).abs()
    spike_mask = diff > (k * scale)

    s_filt = s.copy()
    s_filt[spike_mask] = med[spike_mask]
    return s_filt, spike_mask.fillna(False)

def apply_hampel_to_hs(df: pd.DataFrame, col: str = "HS_sel", window: int = 15, k: float = 2.0):
    """
    Applys the Hampel filter to a DataFrame column (primarily snow height but can be used on other columns).

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing the column to be filtered. 
    col : str, default "HS_sel"
        Name of the column to filter.
    window : int, default 15
        Rolling window length passed to `hampel_filter_series`.
    k : float, default 2.0
        Outlier threshold.

    Returns
    -------
    pandas.DataFrame
        The same DataFrame instance with `df[col]` overwritten by the filtered
        series. Prints a brief summary of how many points were
        replaced and, timestamps.
    """
    if col not in df.columns:
        print(f"apply_hampel_to_hs: '{col}' not found; skipping.")
        return df

    s_filt, mask = hampel_filter_series(df[col], window=window, k=k)
    n = int(mask.sum())
    if n > 0:
        # If we have timestamps, show when we changed values
        ts_col = "TIMESTAMP" if "TIMESTAMP" in df.columns else None
        if ts_col is not None:
            changed_ts = pd.to_datetime(df.loc[mask, ts_col], errors="coerce").dropna().to_list()
            print(f"Hampel: replaced {n} spike(s) in '{col}'. First few timestamps: {changed_ts[:10]}")
        else:
            print(f"Hampel: replaced {n} spike(s) in '{col}'.")
    else:
        print(f"Hampel: no spikes detected in '{col}'.")

    df[col] = s_filt
    return df


def main():
    if os.path.exists(os.path.join(output_folder, output_file_name)):
         raise FileExistsError(f"Output file {output_file_name} already exists in {output_folder}. Please remove or rename it.")

    os.makedirs(output_folder, exist_ok=True)
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    dataframes = []
    for input_csv in csv_files:
        df = standardize_units_and_save_to_csv(input_csv, output_folder, safe_individual_csv)
        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df = merged_df.sort_values("TIMESTAMP")
    merged_df = merged_df.dropna(axis=1, how="all")    # drop columns that are all NaN

    output_path = os.path.join(output_folder, output_file_name)
    merged_df.to_csv(output_path, index=False)
    print(f"{'-' * 43}\nMerged and cleaned file saved to {output_path}\n{'-' * 43}")

if __name__ == "__main__":
    main()