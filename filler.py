import xarray as xr
import os
file_path = "data/input/Bruarjokull/Modis_x_runoff_2008_2018.nc"
ds = xr.open_dataset(file_path)

# Replace NaNs in the 'AL' variable with 0.2
ds["AL"] = ds["AL"].fillna(0.2)

ds.to_netcdf(os.path.join(os.environ["COSIPY_OUTDIR"], "Modis_x_runoff_2008_2018_filled.nc"))