import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path



def compare_mass_balance(
    stake_file_path,           # path to the stake table (data/afh-vj/Bxx.txt)
    ds,                        # xarray.Dataset: cosipy output file
    plot=True, lat_idx=0, lon_idx=0,
    start_year=None, end_year=None,
    variable_name="MB",
):
    
    stake_file_path = str(Path(stake_file_path))
    df = pd.read_csv(
        stake_file_path,
        sep="\t",
        parse_dates=["d1", "d2", "d3"],
        dtype={"ar": "Int64"},
    )

    required = ["ar", "d1", "d2", "d3", "bw_fld", "bs_fld", "ba_fld"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {stake_file_path}: {missing}")

    
    if start_year is not None:
        df = df[df["ar"] >= start_year]
    if end_year is not None:
        df = df[df["ar"] <= end_year]
    def sum_over_period(t1, t2):
        da = ds[variable_name]
        t1 = pd.to_datetime(t1); t2 = pd.to_datetime(t2)
        if set(["time","lat","lon"]).issubset(da.dims):
            sub = da.sel(time=slice(t1, t2)).isel(lat=lat_idx, lon=lon_idx)
        elif set(["time","south_north","west_east"]).issubset(da.dims):
            sub = da.sel(time=slice(t1, t2)).isel(south_north=lat_idx, west_east=lon_idx)
        else:
            spatial = [d for d in da.dims if d != "time"]
            if len(spatial) != 2:
                raise ValueError(f"Cannot infer spatial dims for {variable_name}: {da.dims}")
            sub = da.sel(time=slice(t1, t2)).isel({spatial[0]: lat_idx, spatial[1]: lon_idx})
        return float(sub.sum().values)

    rows = []
    for _, r in df.iterrows():
        d1, d2, d3 = r["d1"], r["d2"], r["d3"]
        if pd.isna(d1) or pd.isna(d2) or pd.isna(d3):
            continue

    
        bw_model = sum_over_period(d1, d2)
        bs_model = sum_over_period(d2, d3)
        ba_model = sum_over_period(d1, d3)

        bw_meas = float(r["bw_fld"]) if pd.notnull(r["bw_fld"]) else np.nan
        bs_meas = float(r["bs_fld"]) if pd.notnull(r["bs_fld"]) else np.nan
        ba_meas = float(r["ba_fld"]) if pd.notnull(r["ba_fld"]) else np.nan

        rows.append({
            "year": int(r["ar"]) if pd.notnull(r["ar"]) else int(pd.to_datetime(d3).year),
            "bw_fld": bw_meas, "bs_fld": bs_meas, "ba_fld": ba_meas,
            "bw_model": bw_model, "bs_model": bs_model, "ba_model": ba_model,
        })

    result_df = pd.DataFrame(rows).dropna(subset=["bw_fld","bs_fld","bw_model","bs_model"])
    if result_df.empty:
        print("No overlapping periods")
        return None
    
    def _pair(y, x):
        y = np.asarray(y, float); x = np.asarray(x, float)
        m = np.isfinite(y) & np.isfinite(x)
        return y[m], x[m]

    def _rmse(y, x):
        y,x = _pair(y,x);  return float(np.sqrt(np.mean((x - y)**2))) if len(y) else np.nan
    def _mae(y, x):
        y,x = _pair(y,x);  return float(np.mean(np.abs(x - y))) if len(y) else np.nan
    def _corr(y, x):
        y,x = _pair(y,x);  return float(np.corrcoef(y, x)[0,1]) if len(y) >= 2 else np.nan
    def _bias(y, x):
        y,x = _pair(y,x);  return float(np.mean(x - y)) if len(y) else np.nan
    def _over(y, x):
        y,x = _pair(y,x);  return float(np.mean(x > y)) if len(y) else np.nan

    bw_rmse, bw_mae, bw_corr = _rmse(result_df["bw_fld"], result_df["bw_model"]), _mae(result_df["bw_fld"], result_df["bw_model"]), _corr(result_df["bw_fld"], result_df["bw_model"])
    bs_rmse, bs_mae, bs_corr = _rmse(result_df["bs_fld"], result_df["bs_model"]), _mae(result_df["bs_fld"], result_df["bs_model"]), _corr(result_df["bs_fld"], result_df["bs_model"])
    ba_rmse, ba_mae, ba_corr = _rmse(result_df["ba_fld"], result_df["ba_model"]), _mae(result_df["ba_fld"], result_df["ba_model"]), _corr(result_df["ba_fld"], result_df["ba_model"])

    bw_bias, bs_bias, ba_bias = _bias(result_df["bw_fld"], result_df["bw_model"]), _bias(result_df["bs_fld"], result_df["bs_model"]), _bias(result_df["ba_fld"], result_df["ba_model"])
    bw_over, bs_over, ba_over = _over(result_df["bw_fld"], result_df["bw_model"]), _over(result_df["bs_fld"], result_df["bs_model"]), _over(result_df["ba_fld"], result_df["ba_model"])

    print("\nEVALUATION METRICS (units: m w.e.)")
    print(f"BW  RMSE: {bw_rmse:.3f}, Corr: {bw_corr:.2f}, Bias: {bw_bias:.3f}")
    print(f"BS  RMSE: {bs_rmse:.3f}, Corr: {bs_corr:.2f}, Bias: {bs_bias:.3f}")
    print(f"BA  RMSE: {ba_rmse:.3f}, Corr: {ba_corr:.2f}, Bias: {ba_bias:.3f}")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        yrs = result_df["year"]
        ax.plot(yrs, result_df["bw_fld"], "o-", label="bw meas",  color="blue")
        ax.plot(yrs, result_df["bs_fld"], "o-", label="bs meas",  color="red")
        ax.plot(yrs, result_df["ba_fld"], "o-", label="ba meas",  color="green")
        ax.plot(yrs, result_df["bw_model"], "o--", label="bw model",  color="blue")
        ax.plot(yrs, result_df["bs_model"], "o--", label="bs model",  color="red")
        ax.plot(yrs, result_df["ba_model"], "o--", label="ba model",  color="green")
        ax.set_xlabel("Year"); ax.set_ylabel("MB (m w.e.)")
        ax.set_title("Seasonal and Annual Mass Balance")
        ax.grid(True, linestyle="--", alpha=0.4); ax.legend(ncol=2)

        
        ax = axes[1]
        ax.scatter(result_df["bw_fld"], result_df["bw_model"], label="Winter MB", c="blue")
        ax.scatter(result_df["bs_fld"], result_df["bs_model"], label="Summer MB", c="red")
        ax.scatter(result_df["ba_fld"], result_df["ba_model"], label="Annual MB", c="green")
        all_vals = pd.concat([
            result_df["bw_fld"], result_df["bw_model"],
            result_df["bs_fld"], result_df["bs_model"],
            result_df["ba_fld"], result_df["ba_model"],
        ], ignore_index=True)
        lo, hi = np.nanpercentile(all_vals, [1, 99]) if len(all_vals) else (0, 1)
        pad = 0.05 * (hi - lo if np.isfinite(hi - lo) else 1.0)
        lims = [float(lo - pad), float(hi + pad)]
        ax.plot(lims, lims, "--k")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Measured (m w.e.)"); ax.set_ylabel("Modeled (m w.e.)")
        ax.set_title("Modeled vs Measured")
        ax.legend(loc="upper left"); ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.show()

    return {
        "bw_rmse": bw_rmse, "bs_rmse": bs_rmse, "ba_rmse": ba_rmse,
        "bw_mae":  bw_mae,  "bs_mae":  bs_mae,  "ba_mae":  ba_mae,
        "bw_corr": bw_corr, "bs_corr": bs_corr, "ba_corr": ba_corr,
        "bw_bias_mean": bw_bias, "bs_bias_mean": bs_bias, "ba_bias_mean": ba_bias,
        "bw_over_frac": bw_over, "bs_over_frac": bs_over, "ba_over_frac": ba_over,
        "df": result_df
    }




def select_nearest_point(cosipy_out, lat=None, lon=None):
    """
    Select the nearest grid point in COSIPY output dataset based on latitude and longitude.

    Parameters:
    cosipy_out (xr.Dataset): COSIPY output.
    lat (float, optional): Latitude to select nearest grid point.
    lon (float, optional): Longitude to select nearest grid point.

    Returns:
    dict: Dictionary with selected lat, lon, and elevation (HGT).
    dict: Selection dictionary for xarray operations.
    """
    if lat is not None and lon is not None:
        sel_dict = dict(lat=lat, lon=lon, method="nearest")
    else:
        sel_dict = dict(lat=cosipy_out.lat[0], lon=cosipy_out.lon[0])

    selected_lat = float(cosipy_out.lat.sel(lat=sel_dict['lat'], method="nearest").values)
    selected_lon = float(cosipy_out.lon.sel(lon=sel_dict['lon'], method="nearest").values)
    hgt = float(cosipy_out["HGT"].sel(**sel_dict).values)

    return {"lat": selected_lat, "lon": selected_lon, "HGT": hgt}, sel_dict


def comp_aws_2_cosipy_output(aws, cosipy_out, readable_name, start_year=None, end_year=None, lat=None, lon=None):
    """
    Plot comparison of AWS data and COSIPY output with optional per-year alignment (for height vars).
    Computes RMSE strictly over the *displayed time window*.
    Inputs are never modified.

    Parameters
    ----------
    aws : pandas.DataFrame
        Must contain TIMESTAMP and the mapped AWS column below.
    cosipy_out : xarray.Dataset
        COSIPY output dataset.
    readable_name : str
        One of: TOTALHEIGHT, SNOWHEIGHT, ALBEDO, TS, MB, RRR, RAIN, SNOWFALL, LWin, LWout
    start_year, end_year : int or None
        Start/end years for filtering/plot window. If only start_year given, plots that single year.
    lat, lon : float or None
        Location for selecting the nearest COSIPY grid point.
    """

 
    variable_dict = {
        "TOTALHEIGHT": {"aws_var": "HS_sel",         "cosipy_var": "TOTALHEIGHT"},
        "SNOWHEIGHT":  {"aws_var": "HS_sel",         "cosipy_var": "SNOWHEIGHT"},
        "ALBEDO":      {"aws_var": "Albedo_acc",     "cosipy_var": "ALBEDO"},
        "TS":          {"aws_var": "Ts",             "cosipy_var": "TS"},
        "MB":          {"aws_var": "nan",            "cosipy_var": "MB"},
        "RRR":         {"aws_var": "TotalPrecipmm",  "cosipy_var": "RRR"},   # both in mm
        "RAIN":        {"aws_var": "Rainfallmweq",   "cosipy_var": "RAIN"},  # AWS m w.e., COSIPY mm
        "SNOWFALL":    {"aws_var": "Snowfallmweq",   "cosipy_var": "SNOWFALL"},
        "LWin":        {"aws_var": "lw_in",          "cosipy_var": "LWin"},
        "LWout":       {"aws_var": "lw_out",         "cosipy_var": "LWout"},
    }
    if readable_name not in variable_dict:
        print(f"Variable '{readable_name}' not found in the mapping.")
        return

    aws_var_name    = variable_dict[readable_name]["aws_var"]
    cosipy_var_name = variable_dict[readable_name]["cosipy_var"]

    
    aws_plot = aws.copy()
    if "nan" not in aws_plot.columns:
        aws_plot["nan"] = np.nan

    location_info, sel_dict = select_nearest_point(cosipy_out, lat, lon)
    selected_lat = location_info["lat"]
    selected_lon = location_info["lon"]
    hgt          = location_info["HGT"]
    cosipy_data  = cosipy_out[cosipy_var_name].sel(**sel_dict).to_dataframe().reset_index()

    
    overall_rmse = None
    yearly_rmse  = []

    if readable_name in ["SNOWHEIGHT", "TOTALHEIGHT"]:
        if "TIMESTAMP" not in aws_plot or aws_var_name not in aws_plot:
            print("Missing TIMESTAMP or required AWS column.")
            return
        aws_plot["Year"] = aws_plot["TIMESTAMP"].dt.year

        # years list
        if start_year is not None and end_year is not None:
            years = range(start_year, end_year + 1)
        elif start_year is not None:
            years = [start_year]
        else:
            years = aws_plot["Year"].dropna().unique()

        aligned = []
        for year in years:
            g = aws_plot[aws_plot["Year"] == year].copy()
            summer_g = g.dropna(subset=[aws_var_name])
            if summer_g.empty:
                continue
            first_date = summer_g["TIMESTAMP"].iloc[0]
            c0 = cosipy_data.loc[cosipy_data["time"] == first_date, cosipy_var_name].values
            if len(c0) > 0:
                adj = c0[0] - summer_g[aws_var_name].iloc[0]
                g.loc[:, aws_var_name] = g[aws_var_name] + adj

                tmp = (
                    pd.merge(g, cosipy_data, left_on="TIMESTAMP", right_on="time", how="inner")
                      .dropna(subset=[aws_var_name, cosipy_var_name])
                )
                if not tmp.empty:
                    rmse_y = float(np.sqrt(mean_squared_error(tmp[aws_var_name], tmp[cosipy_var_name])))
                    yearly_rmse.append((year, rmse_y))
            aligned.append(g)

        aws_aligned = pd.concat(aligned) if aligned else aws_plot.copy()

    elif readable_name == "MB":
        aws_aligned = aws_plot.copy()

    else:
        aws_aligned = aws_plot.copy()

    
    # converting the rain in the aws from mweq to mm
    if readable_name == "RAIN" and aws_var_name in aws_aligned.columns:
        aws_aligned = aws_aligned.copy()
        aws_aligned[aws_var_name] = aws_aligned[aws_var_name] * 1000.0

    
    if start_year is not None and end_year is not None:
        start_date = pd.Timestamp(f"{start_year}-01-01"); end_date = pd.Timestamp(f"{end_year}-12-31")
    elif start_year is not None:
        start_date = pd.Timestamp(f"{start_year}-01-01"); end_date = pd.Timestamp(f"{start_year}-12-31")
    else:
        start_date = end_date = None

    aws_win    = aws_aligned
    cosipy_win = cosipy_data
    if start_date is not None and end_date is not None:
        aws_win    = aws_aligned[(aws_aligned["TIMESTAMP"] >= start_date) & (aws_aligned["TIMESTAMP"] <= end_date)]
        cosipy_win = cosipy_data[(cosipy_data["time"]      >= start_date) & (cosipy_data["time"]      <= end_date)]

    
    if aws_var_name in aws_win.columns:
        merged = (
            pd.merge(aws_win, cosipy_win, left_on="TIMESTAMP", right_on="time", how="inner")
              .dropna(subset=[aws_var_name, cosipy_var_name])
        )
        if not merged.empty:
            overall_rmse = float(np.sqrt(mean_squared_error(merged[aws_var_name], merged[cosipy_var_name])))

   
    legend_loc = f"lat={selected_lat:.3f}, lon={selected_lon:.3f}, HGT={hgt:.1f} m"

    plt.figure(figsize=(12, 6))
    aws_series = aws_win[aws_var_name] if aws_var_name in aws_win.columns else pd.Series(index=aws_win.index, dtype=float)
    if readable_name == "LWout":
        aws_series = -aws_series  # sign convention for plotting only

    plt.plot(aws_win["TIMESTAMP"], aws_series, label=f"{aws_var_name} (AWS)", linewidth=1, alpha=0.7)
    plt.plot(cosipy_win["time"], cosipy_win[cosipy_var_name], label=f"{cosipy_var_name} (COSIPY)", linewidth=1, alpha=0.7)

    
    all_vals = pd.concat([aws_series, cosipy_win[cosipy_var_name]], ignore_index=True)
    if np.isfinite(all_vals.to_numpy()).any():
        plt.ylim(all_vals.min(), all_vals.max())

    
    units = "mm" if readable_name == "RAIN" else cosipy_out[cosipy_var_name].attrs.get("units", "")
    plt.xlabel("Time")
    plt.ylabel(f"{readable_name} ({units})")
    plt.title(f"Comparison AWS input and COSIPY output at {hgt:.1f} m\n(Location: lat={selected_lat:.3f}, lon={selected_lon:.3f})")
    plt.legend(title=legend_loc)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # print metrics
    if readable_name in ["SNOWHEIGHT", "TOTALHEIGHT"]:
        for y, v in yearly_rmse:
            print(f"{y}: RMSE = {v:.4f}")
    if overall_rmse is not None:
        print(f"Overall RMSE of {readable_name}: {overall_rmse:.4f}")

        
def _default_latlon(ds):
        lat = float(np.atleast_1d(ds.lat.values)[0])
        lon = float(np.atleast_1d(ds.lon.values)[0])
        return lat, lon

def _parse_latlon(latlon, ds):
        if latlon is None:
            return _default_latlon(ds)
        if isinstance(latlon, dict):
            if "lat" in latlon and "lon" in latlon:
                return float(latlon["lat"]), float(latlon["lon"])
            raise ValueError("latlon dict must have keys 'lat' and 'lon'.")
        if isinstance(latlon, (list, tuple)) and len(latlon) == 2:
            return float(latlon[0]), float(latlon[1])  # [lat, lon]
        raise ValueError("latlon must be {'lat':..., 'lon':...} or [lat, lon].")

def compare_outputs(dataset1, dataset2=None, dataset3=None, dataset4=None,
                    variable_name=None,
                    latlon1=None, latlon2=None, latlon3=None, latlon4=None,
                    labels=None, start_year=None, end_year=None):
    """
    Compare a specific variable from up to 4 datasets and plot them.

    - If only start_year is given, plot that single calendar year.
    - If start_year and end_year are given, plot the inclusive window.
    - If neither is given, plot the full range.

    """

    if not variable_name:
        raise ValueError("Please provide variable_name")

    vn = variable_name.strip()
    lower = vn.lower()
    use_cum = False
    use_mean = False
    while lower.startswith("cum") or lower.startswith("mean") or lower.startswith("_"):
        if lower.startswith("_"):
            vn = vn[1:]; lower = lower[1:]
            continue
        if lower.startswith("cum"):
            use_cum = True
            vn = vn[3:]; lower = lower[3:]
            if lower.startswith("_"):
                vn = vn[1:]; lower = lower[1:]
            continue
        if lower.startswith("mean"):
            use_mean = True
            vn = vn[4:]; lower = lower[4:]
            if lower.startswith("_"):
                vn = vn[1:]; lower = lower[1:]
            continue
    base_var = vn
    if not base_var:
        raise ValueError("After removing prefixes, variable name is empty.")

    ds_list  = [dataset1, dataset2, dataset3, dataset4]
    ll_list  = [latlon1,  latlon2,  latlon3,  latlon4]

    for i in range(4):
        if ds_list[i] is None and (ll_list[i] is not None):
            if dataset1 is None:
                raise ValueError("No dataset provided to reuse for extra locations.")
            ds_list[i] = dataset1

    # Labels
    default_labels = [f"Dataset {i+1}" for i in range(4)]
    label_list = default_labels if labels is None else (list(labels) + default_labels)[:4]

   
    if start_year is not None:
        sy = int(start_year)
        ey = int(end_year) if end_year is not None else sy
        start_date = pd.Timestamp(f"{sy}-01-01")
        end_date   = pd.Timestamp(f"{ey}-12-31")
    else:
        start_date = end_date = None

    prepared = []
    for i in range(4):
        ds = ds_list[i]
        if ds is None or base_var not in ds:
            continue

        lat, lon = _parse_latlon(ll_list[i], ds)  # helper must exist in your environment
        sel = dict(lat=lat, lon=lon, method="nearest")

        try:
            lat_sel = float(ds.lat.sel(lat=sel["lat"], method="nearest").values)
            lon_sel = float(ds.lon.sel(lon=sel["lon"], method="nearest").values)
        except Exception:
            continue

        hgt = None
        if "HGT" in ds:
            try:
                hgt = float(ds["HGT"].sel(**sel).values)
            except Exception:
                hgt = None

        df = ds[base_var].sel(**sel).to_dataframe().reset_index()
        df["time"] = pd.to_datetime(df["time"])

        
        if start_date is not None:
            df = df[df["time"] >= start_date]
        if end_date is not None:
            df = df[df["time"] <= end_date]

        if df.empty:
            continue

        if use_cum:
            df[base_var] = df[base_var].cumsum()

        prepared.append({"df": df, "lat": lat_sel, "lon": lon_sel, "hgt": hgt, "idx": i})

    if not prepared:
        print("No valid series to plot.")
        return

    
    units = ""
    for ds in ds_list:
        if ds is not None and base_var in ds and hasattr(ds[base_var], "attrs"):
            units = ds[base_var].attrs.get("units", "")
            if units:
                break

    colors = ["blue", "orange", "violet", "green"]

    plt.figure(figsize=(12, 6))
    for item in prepared:
        idx = item["idx"]
        base_label = label_list[idx]
        if item["hgt"] is not None:
            lbl = f"{base_label} ({item['hgt']:.0f}m, {item['lat']:.3f}, {item['lon']:.3f})"
        else:
            lbl = f"{base_label} ({item['lat']:.3f}, {item['lon']:.3f})"
        plt.plot(item["df"]["time"], item["df"][base_var],
                 label=lbl, color=colors[idx], linewidth=.9, alpha=0.8)

    if use_mean:
        for item in prepared:
            idx = item["idx"]
            m = item["df"][base_var].mean()
            plt.axhline(m, color=colors[idx], linestyle='--',
                        label=f"Mean {label_list[idx]}: {m:.2f}")

    
    if start_year is not None:
        window_txt = f" ({int(start_year)}" + (f"–{int(end_year)}" if end_year is not None else "") + ")"
    else:
        window_txt = ""

    title_var = base_var
    if use_cum and use_mean:
        title_var = f"cum(mean({base_var}))"
    elif use_cum:
        title_var = f"cum({base_var})"
    elif use_mean:
        title_var = f"mean({base_var})"

    plt.xlabel("Time", fontsize=12)
    plt.ylabel(f"{base_var}" + (f" ({units})" if units else ""), fontsize=12)
    plt.title(f"Comparison of {title_var}{window_txt}", fontsize=14)
    plt.legend(fontsize=10, ncol=2)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def compare_variables(dataset1, variable_name1, variable_name2, year=None, lat=None, lon=None):
    """
    Compare two variables from a dataset and plot them with dual y-axes, with an option to filter by year and location.

    Parameters:
    dataset1 (xr.Dataset): Dataset containing the variables.
    variable_name1 (str): First variable name to compare.
    variable_name2 (str): Second variable name to compare.
    year (int, optional): Year to filter the data. If None, the full range is used.
    lat (float, optional): Latitude to select nearest grid point.
    lon (float, optional): Longitude to select nearest grid point.

    Returns:
    None
    """
   
    if lat is not None and lon is not None:
        sel_dict = dict(lat=lat, lon=lon, method="nearest")
        
    else:
        sel_dict = dict(lat=dataset1.lat[0], lon=dataset1.lon[0])

    
    selected_lat = float(dataset1.lat.sel(lat=sel_dict['lat'], method="nearest").values)
    selected_lon = float(dataset1.lon.sel(lon=sel_dict['lon'], method="nearest").values)

    
    hgt = float(dataset1["HGT"].sel(**sel_dict).values)

    if variable_name1 in dataset1:
        data1 = dataset1[variable_name1].sel(**sel_dict).to_dataframe().reset_index()
    else:
        print(f"Variable '{variable_name1}' not found in the dataset.")
        return

    if variable_name2 in dataset1:
        data2 = dataset1[variable_name2].sel(**sel_dict).to_dataframe().reset_index()
    else:
        print(f"Variable '{variable_name2}' not found in the dataset.")
        return

    
    if year is not None:
        data1["time"] = pd.to_datetime(data1["time"])
        data2["time"] = pd.to_datetime(data2["time"])
        data1 = data1[data1["time"].dt.year == year]
        data2 = data2[data2["time"].dt.year == year]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color1 = "blue"
    color2 = "orange"

    ax1.set_xlabel("Time", fontsize=12)
    ax1.grid(True, axis="both", linestyle="--", alpha=0.5)
    ax1.set_ylabel(f"{variable_name1} ({dataset1[variable_name1].attrs.get('units', '')})", color=color1, fontsize=12)
    ax1.plot(data1["time"], data1[variable_name1], label=f"{variable_name1}", color=color1, linewidth=0.9, alpha=0.8)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.set_ylabel(f"{variable_name2} ({dataset1[variable_name2].attrs.get('units', '')})", color=color2, fontsize=12)
    ax2.plot(data2["time"], data2[variable_name2], label=f"{variable_name2}", color=color2, linewidth=0.9, alpha=0.8)
    ax2.tick_params(axis='y')

    # Final plot title with variable names, elevation, and location
    plt.title(f'Comparison of {variable_name1} and {variable_name2} at {hgt:.1f} m\n(Location: lat={selected_lat:.3f}, lon={selected_lon:.3f})', fontsize=14)
    fig.tight_layout()
    plt.show()

