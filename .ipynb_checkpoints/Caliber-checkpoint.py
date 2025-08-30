
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import mean_squared_error, r2_score

# ==============================
# 1) SUMMARIZE COSIPY FOLDER
# ==============================

def summarize_cosipy_folder(
    folder,
    aws_csv_map,
    stake_map=None,
    attr_keys=None,
    include_all_attrs: bool = True,
    skip_first_years: int = 2,   # <— NEW: number of initial years to skip in metrics
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per NetCDF file. The first `skip_first_years`
    of each dataset's time axis are excluded from RMSE/R² and MB metrics.
    """
    folder = Path(folder)

    # ---------- helpers (unchanged) ----------
    def _station_from_name(name: str):
        stem = Path(name).stem.upper()
        first = stem.split("_")[0]
        if first in {"B10", "B13", "B16"}:
            return first
        m = re.match(r"(B1[036])", stem)
        return m.group(1) if m else None

    def _to_naive(ts):
        s = pd.to_datetime(ts, errors="coerce")
        try:
            if getattr(s.dt, "tz", None) is not None:
                s = s.dt.tz_localize(None)
        except Exception:
            pass
        return s

    def _series_from_var(ds, var):
        if var not in ds.variables or "time" not in ds.coords:
            return pd.DataFrame(columns=["TIMESTAMP", "model"])
        da = ds[var]
        for d in list(da.dims):
            if d != "time":
                da = da.mean(dim=d, skipna=True)
        df = da.to_dataframe().reset_index()
        df = df.rename(columns={"time": "TIMESTAMP", var: "model"})
        df["TIMESTAMP"] = _to_naive(df["TIMESTAMP"])
        return df.dropna(subset=["TIMESTAMP"]).sort_values("TIMESTAMP")

    def _albedo_metrics_exact(ds, aws_df, aws_col="Albedo_acc", model_var="ALBEDO", clip=(0, 1)):
        if aws_df is None or aws_col not in aws_df.columns:
            return np.nan, np.nan, 0
        model = _series_from_var(ds, model_var)
        if model.empty:
            return np.nan, np.nan, 0

        a = aws_df.copy()
        a["TIMESTAMP"] = _to_naive(a["TIMESTAMP"])
        a[aws_col] = pd.to_numeric(a[aws_col], errors="coerce")
        if clip is not None:
            lo, hi = clip
            a[aws_col] = a[aws_col].clip(lo, hi)
        a = a.dropna(subset=["TIMESTAMP", aws_col]).sort_values("TIMESTAMP")
        if a.empty:
            return np.nan, np.nan, 0

        tmin, tmax = model["TIMESTAMP"].min(), model["TIMESTAMP"].max()
        a = a[(a["TIMESTAMP"] >= tmin) & (a["TIMESTAMP"] <= tmax)]
        if a.empty:
            return np.nan, np.nan, 0

        merged = pd.merge(a[["TIMESTAMP", aws_col]], model, on="TIMESTAMP", how="inner").dropna()
        if merged.empty:
            return np.nan, np.nan, 0

        rmse = float(np.sqrt(mean_squared_error(merged[aws_col], merged["model"])))
        r2 = float(r2_score(merged[aws_col], merged["model"]))
        return rmse, r2, int(len(merged))

    def _totalheight_metrics_exact_offset(ds, aws_df):
        if aws_df is None:
            return np.nan, np.nan, 0
        model = _series_from_var(ds, "TOTALHEIGHT")
        if model.empty:
            return np.nan, np.nan, 0

        for hcol in ("HS_sel", "TOTALHEIGHT"):
            if hcol in aws_df.columns:
                break
        else:
            return np.nan, np.nan, 0

        a = aws_df.copy()
        a["TIMESTAMP"] = _to_naive(a["TIMESTAMP"])
        a[hcol] = pd.to_numeric(a[hcol], errors="coerce")
        a = a.dropna(subset=["TIMESTAMP", hcol]).sort_values("TIMESTAMP")
        if a.empty:
            return np.nan, np.nan, 0

        tmin, tmax = model["TIMESTAMP"].min(), model["TIMESTAMP"].max()
        a = a[(a["TIMESTAMP"] >= tmin) & (a["TIMESTAMP"] <= tmax)]
        if a.empty:
            return np.nan, np.nan, 0

        a["Year"] = a["TIMESTAMP"].dt.year
        pairs = []
        for y in sorted(a["Year"].unique()):
            g = a[a["Year"] == y].copy()
            if g.empty:
                continue
            first_date = g["TIMESTAMP"].iloc[0]
            m0 = model.loc[model["TIMESTAMP"] == first_date, "model"].values
            if len(m0) == 0:
                continue
            offset = float(m0[0] - g[hcol].iloc[0])
            g.loc[:, hcol] = g[hcol] + offset

            m_y = model[(model["TIMESTAMP"] >= g["TIMESTAMP"].min()) &
                        (model["TIMESTAMP"] <= g["TIMESTAMP"].max())]
            merged = pd.merge(g[["TIMESTAMP", hcol]], m_y, on="TIMESTAMP", how="inner").dropna()
            if not merged.empty:
                merged = merged.rename(columns={hcol: "aws"})
                pairs.append(merged)

        if not pairs:
            return np.nan, np.nan, 0
        both = pd.concat(pairs, ignore_index=True)
        rmse = float(np.sqrt(mean_squared_error(both["aws"], both["model"])))
        r2 = float(r2_score(both["aws"], both["model"]))
        return rmse, r2, int(len(both))

    def _sum_over_period(ds, var, d1, d2):
        if var not in ds or "time" not in ds[var].coords:
            return np.nan
        da = ds[var]
        for d in list(da.dims):
            if d not in ("time",):
                da = da.mean(dim=d, skipna=True)
        t = pd.to_datetime(da["time"].values)
        mask = (t >= pd.Timestamp(d1)) & (t <= pd.Timestamp(d2))
        if not mask.any():
            return np.nan
        return float(da.sel(time=mask).sum().values)

    def _mb_metrics(ds, stake_df, var="MB"):
        if stake_df is None or stake_df.empty or var not in ds.variables:
            return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        rows = []
        for _, r in stake_df.iterrows():
            d1, d2, d3 = r.get("d1"), r.get("d2"), r.get("d3")
            if pd.isnull(d1) or pd.isnull(d2) or pd.isnull(d3):
                continue
            bw_m = _sum_over_period(ds, var, d1, d2)
            bs_m = _sum_over_period(ds, var, d2, d3)
            rows.append({
                "bw_fld": float(r["bw_fld"]) if pd.notnull(r["bw_fld"]) else np.nan,
                "bs_fld": float(r["bs_fld"]) if pd.notnull(r["bs_fld"]) else np.nan,
                "bw_model": bw_m,
                "bs_model": bs_m,
            })
        dfm = pd.DataFrame(rows).dropna()
        if dfm.empty:
            return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        dfm["ba_fld"] = dfm["bw_fld"] + dfm["bs_fld"]
        dfm["ba_model"] = dfm["bw_model"] + dfm["bs_model"]

        def _rmse_r2(ytrue, ypred):
            return (float(np.sqrt(mean_squared_error(ytrue, ypred))),
                    float(r2_score(ytrue, ypred)) if len(ytrue) > 1 else np.nan)

        bw_rmse, bw_r2 = _rmse_r2(dfm["bw_fld"], dfm["bw_model"])
        bs_rmse, bs_r2 = _rmse_r2(dfm["bs_fld"], dfm["bs_model"])
        ba_rmse, ba_r2 = _rmse_r2(dfm["ba_fld"], dfm["ba_model"])
        return (bw_rmse, bw_r2, bs_rmse, bs_r2, ba_rmse, ba_r2)

    def _coerce_scalar(v):
        try:
            import numpy as _np
            if isinstance(v, (_np.generic,)):
                return v.item()
        except Exception:
            pass
        return v

    def _extract_attrs(ds, attr_keys, include_all):
        if include_all:
            out = {}
            for k, v in ds.attrs.items():
                out[str(k)] = _coerce_scalar(v)
            return out
        if not attr_keys:
            return {}
        low_map = {k.lower(): k for k in ds.attrs.keys()}
        out = {}
        for req in attr_keys:
            k0 = low_map.get(req.lower())
            if k0 is not None:
                out[req] = _coerce_scalar(ds.attrs[k0])
            else:
                picked = None
                for k in ds.attrs.keys():
                    if req.lower() in k.lower():
                        picked = k
                        break
                out[req] = _coerce_scalar(ds.attrs[picked]) if picked else np.nan
        return out

    # ---------- main loop (trim first N years) ----------
    rows = []
    files = list(folder.rglob("*.nc")) + list(folder.rglob("*.nc4"))
    for nc in sorted(files):
        station = _station_from_name(nc.name)
        if not station or station not in aws_csv_map:
            continue

        aws = aws_csv_map[station].copy()
        if "TIMESTAMP" not in aws.columns:
            continue
        aws["TIMESTAMP"] = _to_naive(aws["TIMESTAMP"])

        stake_df = None
        if stake_map and station in stake_map:
            stake_df = stake_map[station].copy()
            for c in ("d1", "d2", "d3"):
                if c in stake_df.columns and not pd.api.types.is_datetime64_any_dtype(stake_df[c]):
                    stake_df[c] = pd.to_datetime(stake_df[c], errors="coerce")

        try:
            with xr.open_dataset(nc) as ds:
                ds_use = ds
                aws_use = aws

                if "time" in ds.coords:
                    tvals = pd.to_datetime(ds["time"].values)
                    if len(tvals) == 0:
                        continue
                    tmin, tmax = tvals.min(), tvals.max()
                    trim_start = (pd.Timestamp(tmin) + pd.DateOffset(years=skip_first_years)).to_pydatetime()
                    # Trim dataset and AWS to [trim_start, tmax]
                    ds_use = ds.sel(time=slice(trim_start, None))
                    aws_use = aws[(aws["TIMESTAMP"] >= trim_start) & (aws["TIMESTAMP"] <= tmax)].copy()
                else:
                    # No time axis; nothing to trim
                    aws_use = aws

                # ---- metrics on trimmed data ----
                alb_rmse, alb_r2, n_alb = _albedo_metrics_exact(ds_use, aws_use, aws_col="Albedo_acc",
                                                                model_var="ALBEDO", clip=(0, 1))
                th_rmse, th_r2, n_th = _totalheight_metrics_exact_offset(ds_use, aws_use)
                bw_rmse, bw_r2, bs_rmse, bs_r2, ba_rmse, ba_r2 = _mb_metrics(ds_use, stake_df)

                # Attributes still taken from full dataset (global attrs are static)
                attr_vals = _extract_attrs(ds, attr_keys, include_all_attrs)

        except Exception:
            continue

        rows.append({
            "station": station,
            "file": str(nc),
            "rmse_albedo": alb_rmse, "r2_albedo": alb_r2, "n_albedo": n_alb,
            "rmse_totalheight": th_rmse, "r2_totalheight": th_r2, "n_totalheight": n_th,
            "bw_rmse": bw_rmse, "bw_r2": bw_r2,
            "bs_rmse": bs_rmse, "bs_r2": bs_r2,
            "ba_rmse": ba_rmse, "ba_r2": ba_r2,
            **attr_vals
        })

    return pd.DataFrame(rows)



aws_csv_map = {
    #"B10": pd.read_csv("Cosipy_jonas/cosipy/data/input/Bruarjokull/B10_final_all.csv", parse_dates=["TIMESTAMP"]),
    #"B13": pd.read_csv("Cosipy_jonas/cosipy/data/input/Bruarjokull/B13_final_all.csv", parse_dates=["TIMESTAMP"]),
    "B13": pd.read_csv("data/input/Bruarjokull/B13_final_all.csv", parse_dates=["TIMESTAMP"]),
}
stake_map = {
    #"B10": pd.read_csv("Cosipy_jonas/cosipy/data/afh-vj/B10.txt", sep="\t", parse_dates=["d1","d2","d3"]),
    #"B13": pd.read_csv("Cosipy_jonas/cosipy/data/afh-vj/B13.txt", sep="\t", parse_dates=["d1","d2","d3"]),
    "B13": pd.read_csv("data/afh-vj/B13.txt", sep="\t", parse_dates=["d1","d2","d3"]),
}

# 2) Run
df = summarize_cosipy_folder(
    folder="../../Calibration_files/B13_rainsnow/output",
    aws_csv_map=aws_csv_map,
    stake_map=stake_map,
    attr_keys=["Albedo_fresh_snow", "Albedo_firn", "Albedo_ice", "Albedo_mod_snow_aging", "Minimum_snowfall", "Multiplication_factor_for_RRR_or_SNOWFALL"]
)


# 3) Save
df.to_csv("../../Calibration_files/B13_Summarys/B13_rainsnow.csv", index=False)
################################################################




################################################################

aws_csv_map = {
    #"B10": pd.read_csv("Cosipy_jonas/cosipy/data/input/Bruarjokull/B10_final_all.csv", parse_dates=["TIMESTAMP"]),
    #"B13": pd.read_csv("Cosipy_jonas/cosipy/data/input/Bruarjokull/B13_final_all.csv", parse_dates=["TIMESTAMP"]),
    "B13": pd.read_csv("data/input/Bruarjokull/B13_final_all.csv", parse_dates=["TIMESTAMP"]),
}
stake_map = {
    #"B10": pd.read_csv("Cosipy_jonas/cosipy/data/afh-vj/B10.txt", sep="\t", parse_dates=["d1","d2","d3"]),
    #"B13": pd.read_csv("Cosipy_jonas/cosipy/data/afh-vj/B13.txt", sep="\t", parse_dates=["d1","d2","d3"]),
    "B13": pd.read_csv("data/afh-vj/B13.txt", sep="\t", parse_dates=["d1","d2","d3"]),
}

# 2) Run
df = summarize_cosipy_folder(
    folder="../../Calibration_files/B13_stageA_albedo",
    aws_csv_map=aws_csv_map,
    stake_map=stake_map,
    attr_keys=["Albedo_fresh_snow", "Albedo_firn", "Albedo_ice", "Albedo_mod_snow_aging", "Minimum_snowfall", "Multiplication_factor_for_RRR_or_SNOWFALL"]
)


# 3) Save
df.to_csv("../../Calibration_files/B13_Summarys/B13_stageA.csv", index=False)
##############################################################################
################################################################

aws_csv_map = {
    #"B10": pd.read_csv("Cosipy_jonas/cosipy/data/input/Bruarjokull/B10_final_all.csv", parse_dates=["TIMESTAMP"]),
    #"B13": pd.read_csv("Cosipy_jonas/cosipy/data/input/Bruarjokull/B13_final_all.csv", parse_dates=["TIMESTAMP"]),
    "B13": pd.read_csv("data/input/Bruarjokull/B13_final_all.csv", parse_dates=["TIMESTAMP"]),
}
stake_map = {
    #"B10": pd.read_csv("Cosipy_jonas/cosipy/data/afh-vj/B10.txt", sep="\t", parse_dates=["d1","d2","d3"]),
    #"B13": pd.read_csv("Cosipy_jonas/cosipy/data/afh-vj/B13.txt", sep="\t", parse_dates=["d1","d2","d3"]),
    "B13": pd.read_csv("data/afh-vj/B13.txt", sep="\t", parse_dates=["d1","d2","d3"]),
}

# 2) Run
df = summarize_cosipy_folder(
    folder="../../Calibration_files/B13_stageB",
    aws_csv_map=aws_csv_map,
    stake_map=stake_map,
    attr_keys=["Albedo_fresh_snow", "Albedo_firn", "Albedo_ice", "Albedo_mod_snow_aging", "Minimum_snowfall", "Multiplication_factor_for_RRR_or_SNOWFALL"]
)


# 3) Save
df.to_csv("../../Calibration_files/B13_Summarys/B13_stageB.csv", index=False)
##############################################################################
################################################################

aws_csv_map = {
    #"B10": pd.read_csv("Cosipy_jonas/cosipy/data/input/Bruarjokull/B10_final_all.csv", parse_dates=["TIMESTAMP"]),
    #"B13": pd.read_csv("Cosipy_jonas/cosipy/data/input/Bruarjokull/B13_final_all.csv", parse_dates=["TIMESTAMP"]),
    "B13": pd.read_csv("data/input/Bruarjokull/B13_final_all.csv", parse_dates=["TIMESTAMP"]),
}
stake_map = {
    #"B10": pd.read_csv("Cosipy_jonas/cosipy/data/afh-vj/B10.txt", sep="\t", parse_dates=["d1","d2","d3"]),
    #"B13": pd.read_csv("Cosipy_jonas/cosipy/data/afh-vj/B13.txt", sep="\t", parse_dates=["d1","d2","d3"]),
    "B13": pd.read_csv("data/afh-vj/B13.txt", sep="\t", parse_dates=["d1","d2","d3"]),
}

# 2) Run
df = summarize_cosipy_folder(
    folder="../../Calibration_files/B13_stageC",
    aws_csv_map=aws_csv_map,
    stake_map=stake_map,
    attr_keys=["Albedo_fresh_snow", "Albedo_firn", "Albedo_ice", "Albedo_mod_snow_aging", "Minimum_snowfall", "Multiplication_factor_for_RRR_or_SNOWFALL"]
)


# 3) Save
df.to_csv("../../Calibration_files/B13_Summarys/B13_stageC.csv", index=False)
##############################################################################
################################################################

aws_csv_map = {
    #"B10": pd.read_csv("Cosipy_jonas/cosipy/data/input/Bruarjokull/B10_final_all.csv", parse_dates=["TIMESTAMP"]),
    #"B13": pd.read_csv("Cosipy_jonas/cosipy/data/input/Bruarjokull/B13_final_all.csv", parse_dates=["TIMESTAMP"]),
    "B13": pd.read_csv("data/input/Bruarjokull/B13_final_all.csv", parse_dates=["TIMESTAMP"]),
}
stake_map = {
    #"B10": pd.read_csv("Cosipy_jonas/cosipy/data/afh-vj/B10.txt", sep="\t", parse_dates=["d1","d2","d3"]),
    #"B13": pd.read_csv("Cosipy_jonas/cosipy/data/afh-vj/B13.txt", sep="\t", parse_dates=["d1","d2","d3"]),
    "B13": pd.read_csv("data/afh-vj/B13.txt", sep="\t", parse_dates=["d1","d2","d3"]),
}

# 2) Run
df = summarize_cosipy_folder(
    folder="../../Calibration_files/B13_stagefinal",
    aws_csv_map=aws_csv_map,
    stake_map=stake_map,
    attr_keys=["Albedo_fresh_snow", "Albedo_firn", "Albedo_ice", "Albedo_mod_snow_aging", "Minimum_snowfall", "Multiplication_factor_for_RRR_or_SNOWFALL"]
)


# 3) Save
df.to_csv("../../Calibration_files/B13_Summarys/B13_stagefinal.csv", index=False)
##############################################################################


"""
aws_csv_map = {
    #"B10": pd.read_csv("Cosipy_jonas/cosipy/data/input/Bruarjokull/B10_final_all.csv", parse_dates=["TIMESTAMP"]),
    "B13": pd.read_csv("data/input/Bruarjokull/B13_final_all.csv", parse_dates=["TIMESTAMP"]),
    #"B16": pd.read_csv("data/input/Bruarjokull/B16_final_all.csv", parse_dates=["TIMESTAMP"]),
}
stake_map = {
    #"B10": pd.read_csv("Cosipy_jonas/cosipy/data/afh-vj/B10.txt", sep="\t", parse_dates=["d1","d2","d3"]),
    "B13": pd.read_csv("data/afh-vj/B13.txt", sep="\t", parse_dates=["d1","d2","d3"]),
    #"B16": pd.read_csv("data/afh-vj/B16.txt", sep="\t", parse_dates=["d1","d2","d3"]),
}

# 2) Run
df = summarize_cosipy_folder(
    folder="B13_rainsnow/output",
    aws_csv_map=aws_csv_map,
    stake_map=stake_map,
    attr_keys=["Albedo_fresh_snow", "Albedo_firn", "Albedo_ice", "Albedo_mod_snow_aging", "Minimum_snowfall", "Multiplication_factor_for_RRR_or_SNOWFALL"]
)


# 3) Save
df.to_csv("Summarys/B13_rainsnow_summary.csv", index=False)
########################################



##########################################
aws_csv_map = {
    "B10": pd.read_csv("data/input/Bruarjokull/B10_final_all.csv", parse_dates=["TIMESTAMP"]),
    #"B13": pd.read_csv("data/input/Bruarjokull/B13_final_all.csv", parse_dates=["TIMESTAMP"]),
    #"B16": pd.read_csv("data/input/Bruarjokull/B16_final_all.csv", parse_dates=["TIMESTAMP"]),
}
stake_map = {
    "B10": pd.read_csv("data/afh-vj/B10.txt", sep="\t", parse_dates=["d1","d2","d3"]),
    #"B13": pd.read_csv("data/afh-vj/B13.txt", sep="\t", parse_dates=["d1","d2","d3"]),
    #"B16": pd.read_csv("data/afh-vj/B16.txt", sep="\t", parse_dates=["d1","d2","d3"]),
}

# 2) Run
df = summarize_cosipy_folder(
    folder="B10_rainsnow/output",
    aws_csv_map=aws_csv_map,
    stake_map=stake_map,
    attr_keys=["Albedo_fresh_snow", "Albedo_firn", "Albedo_ice", "Albedo_mod_snow_aging", "Minimum_snowfall", "Multiplication_factor_for_RRR_or_SNOWFALL"]
)


# 3) Save
df.to_csv("Summarys/B10_rainsnow_summary.csv", index=False)
"""