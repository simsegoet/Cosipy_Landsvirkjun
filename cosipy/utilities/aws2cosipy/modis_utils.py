import os
import re
import numpy as np
import xarray as xr
from glob import glob
from scipy.interpolate import griddata



def crop_and_regrid_modis(modis_file, lat_min, lat_max, lon_min, lon_max, 
                          target_lat, target_lon, var="R_median_filter_albe_GFD", method="nearest"):
    """
    Crop a MODIS (by Andri) dataset to a lat/lon bounding box and regrid to a target lat–lon grid.

    Parameters
    ----------
    modis_file : str or os.PathLike
        Path to the MODIS NetCDF/GRIB file readable by xarray.
    lat_min, lat_max : float
        Latitude bounds of the crop region in degrees North.
    lon_min, lon_max : float
        Longitude bounds of the crop region in degrees East (negative for West).
    target_lat : array-like (1D)
        Target latitude coordinates (degrees). These define the output DataArray's `lat` axis.
    target_lon : array-like (1D)
        Target longitude coordinates (degrees). These define the output DataArray's `lon` axis.
    var : str, optional
        Name of the 2D variable (over `y`, `x`) to extract and regrid. Default
        `"R_median_filter_albe_GFD"`.
    method : {"nearest", "linear", "cubic"}, optional
        Interpolation method passed to `scipy.interpolate.griddata`. Default `"nearest"`.

    Returns
    -------
    xarray.DataArray
        Interpolated field with dimensions `("lat", "lon")` and coordinates
        `{"lat": target_lat, "lon": target_lon}`. Values are scaled to [0, 1]
        (input assumed in percent).

    """
    
    ds = xr.open_dataset(modis_file)
    ds = ds.set_coords(["lat", "lon"])

    mask = (
        (ds["lat"] >= lat_min) & (ds["lat"] <= lat_max) &
        (ds["lon"] >= lon_min) & (ds["lon"] <= lon_max)
    )
    ds = ds.where(mask)

    valid_mask = ~ds[var].isnull()
    valid_y = valid_mask.any(dim="x")
    valid_x = valid_mask.any(dim="y")
    y_min = int(valid_y.argmax())
    y_max = int(valid_y[::-1].argmax().values)
    x_min = int(valid_x.argmax())
    x_max = int(valid_x[::-1].argmax().values)

    cropped = ds.isel(
        y=slice(y_min, ds.dims["y"] - y_max),
        x=slice(x_min, ds.dims["x"] - x_max)
    )

    lat2d = cropped['lat'].values
    lon2d = cropped['lon'].values
    values = cropped[var].values / 100      # modis data in % bot 0-1 needed

    points = np.column_stack((lat2d.ravel(), lon2d.ravel()))
    data_flat = values.ravel()

    lon_grid, lat_grid = np.meshgrid(target_lon, target_lat)

    data_interp = griddata(points, data_flat, (lat_grid, lon_grid), method=method)

    return xr.DataArray(
        data_interp,
        coords={"lat": target_lat, "lon": target_lon},
        dims=["lat", "lon"]
    )


def build_hourly_modis_overlay_from_folder(
    modis_folder,
    input_aws_path=None,
    input_aws_dataset=None,
    lat_min=None, lat_max=None, lon_min=None, lon_max=None, 
    var="R_median_filter_albe_GFD",
    method="nearest",
    start_time=None,
    end_time=None
):
    if input_aws_dataset is not None:
        aws_ds = input_aws_dataset
    elif input_aws_path is not None:
        aws_ds = xr.open_dataset(input_aws_path)
    else:
        raise ValueError("Either input_aws_path or input_aws_dataset must be provided")

    # target grid
    target_lat = np.atleast_1d(aws_ds.lat.values)
    target_lon = np.atleast_1d(aws_ds.lon.values)
    is_point = (target_lat.size == 1) and (target_lon.size == 1)
    if is_point:
        lat0 = float(target_lat[0])
        lon0 = float(target_lon[0])

    print(f"Using MODIS bounding box: lat {lat_min} to {lat_max}, lon {lon_min} to {lon_max}")
    print("Mode:", "point (nearest pixel)" if is_point else "grid (regrid)")

    times = aws_ds.time.values
    if start_time:
        times = times[times >= np.datetime64(start_time)]
    if end_time:
        times = times[times <= np.datetime64(end_time)]
    if len(times) == 0:
        raise ValueError("No matching time steps in AWS dataset for given start/end range.")

    # load daily modis files
    modis_files = sorted(glob(os.path.join(modis_folder, "*.nc")))
    modis_dict = {}

    for file in modis_files:
        m = re.search(r'(\d{8})', os.path.basename(file))
        if not m:
            continue
        date_str = m.group(1)  # YYYYMMDD
        date = np.datetime64(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}")

        if start_time and date < np.datetime64(start_time):
            continue
        if end_time and date > np.datetime64(end_time):
            continue

        try:
            if is_point:
                
                with xr.open_dataset(file) as ds_mod:
                    if var not in ds_mod:
                        raise KeyError(f"Variable '{var}' not found in {file}")
                    lat2d = ds_mod["lat"].values
                    lon2d = ds_mod["lon"].values
                    vals  = np.asarray(ds_mod[var].values)
                    vals  = np.squeeze(vals)  # ensure 2D (y,x)

                    
                    valid = np.isfinite(vals)
                    if not np.any(valid):
                        raise ValueError("All-NaN MODIS values")

                    w = np.cos(np.deg2rad(lat0))
                    d2 = (lat2d - lat0)**2 + ((lon2d - lon0)*w)**2
                    d2 = np.where(valid, d2, np.inf)
                    iy, ix = np.unravel_index(np.argmin(d2), d2.shape)

                    val = float(vals[iy, ix])
                    if np.isfinite(val) and val > 1.01:
                        val /= 100.0

                    modis_dict[date] = np.array([[val]], dtype=float)

                print(f"{file} nearest-pixel value used (valid={np.isfinite(val)})")

            else:
                modis_regridded = crop_and_regrid_modis(
                    file, lat_min, lat_max, lon_min, lon_max,
                    target_lat, target_lon, var, method
                )
                print(f"{file} has valid data: {np.any(~np.isnan(modis_regridded.values))}")
                modis_dict[date] = modis_regridded.values

        except Exception as e:
            print(f"Skipping {file}: {e}")

    # forward filling each days value for 24 hours
    hourly_data = []
    latest = None
    remaining = 0  # hours remaining from the latest daily value

    for t in times:
        day = np.datetime64(str(t)[:10])
        if day in modis_dict:
            latest = modis_dict[day]
            remaining = 24

        if latest is not None and remaining > 0:
            hourly_data.append(latest)
            remaining -= 1
        else:
            hourly_data.append(np.full((target_lat.size, target_lon.size), np.nan))

    data = np.stack(hourly_data)  # (time, lat, lon)
    result_da = xr.DataArray(
        data=data,
        coords={"time": times, "lat": target_lat, "lon": target_lon},
        dims=("time", "lat", "lon")
    )

    
    result_da = result_da.fillna(9999)

    
    combined_ds = aws_ds.sel(time=times).copy()
    combined_ds["ALBEDO"] = result_da

    print("MODIS integration complete.")
    print("MODIS lat range:", float(target_lat.min()), "to", float(target_lat.max()))
    print("MODIS lon range:", float(target_lon.min()), "to", float(target_lon.max()))

    return combined_ds