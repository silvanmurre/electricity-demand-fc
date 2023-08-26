import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from datetime import datetime
from entsoe import EntsoePandasClient
from herbie import FastHerbie
from pathlib import Path
from shapely.geometry import Point


# TODO: Fully customizable for different countries and maps (resolutions)
def get_raster_points_inside_nl(plot=False, raster_size=0.25):
    shpfilename = shpreader.natural_earth(resolution="110m", category="cultural", name="admin_0_countries")
    countries = gpd.read_file(shpfilename)
    netherlands = countries[countries.NAME == "Netherlands"]

    # Determine bounding box of the Netherlands and align it to raster_size
    minx, miny, maxx, maxy = netherlands.total_bounds
    minx = np.floor(minx / raster_size) * raster_size
    miny = np.floor(miny / raster_size) * raster_size
    maxx = np.ceil(maxx / raster_size) * raster_size
    maxy = np.ceil(maxy / raster_size) * raster_size

    x = np.arange(minx, maxx + raster_size, raster_size)
    y = np.arange(miny, maxy + raster_size, raster_size)
    grid_points = [Point(i, j) for i in x for j in y]

    # Convert grid points to GeoDataFrame for easier plotting
    grid_gdf = gpd.GeoDataFrame(geometry=grid_points)

    # TODO: 50m and 10m include Dutch Carribean islands, find way to only include mainland
    # Maybe through biggest polygon, different shapefile or manual bbox borders

    # Filter points inside the Netherlands
    netherlands_shape = netherlands.geometry.iloc[0]
    inside_points = grid_gdf[grid_gdf.geometry.within(netherlands_shape)]

    if plot:
        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        ax.add_geometries(
            netherlands.geometry,
            crs=ccrs.PlateCarree(),
            edgecolor="black",
            facecolor="none",
        )
        grid_gdf.plot(ax=ax, markersize=10, color="red", transform=ccrs.PlateCarree())
        inside_points.plot(ax=ax, markersize=10, color="blue", transform=ccrs.PlateCarree())

        ax.set_extent([minx, maxx, miny, maxy], ccrs.PlateCarree())
        ax.coastlines(resolution="110m")
        plt.show()

    return inside_points.reset_index(drop=True)


def round_down_to_nearest_six_hour(dt):
    # Extract the number of hours and round down to the nearest 6-hourly interval
    new_hour = (dt.hour // 6) * 6

    # Create a new datetime object rounded down to the nearest 6-hour interval
    rounded_dt = dt.replace(hour=new_hour, minute=0, second=0, microsecond=0)

    return rounded_dt


def get_gfs_data(start_date, end_date):
    models_dir = Path("data")
    models_dir.mkdir(parents=True, exist_ok=True)

    start_date = round_down_to_nearest_six_hour(start_date)
    end_date = round_down_to_nearest_six_hour(end_date)
    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="6H",
    )

    H = FastHerbie(
        DATES=dates,
        fxx=range(0, 6, 3),
        model="gfs",
        product="pgrb2.0p50",
        save_dir="data",
        max_threads=16,
        verbose=True,
    )
    # Only retrieve the temperature as downloads are taking too long already
    ds = H.xarray(":TMP:2 m above ground")
    return ds


def extract_gfs_data(
    points: gpd.GeoDataFrame,
    dataset,
    variable,
    start_date: datetime,
    end_date: datetime,
):
    index = pd.date_range(
        start=start_date,
        end=end_date,
        freq="3H",
    )

    data = {}
    for idx, row in points.iterrows():
        point = row.geometry
        lat, lon = point.y, point.x

        # Check if latitude and longitude are within the bounds of the xarray Dataset
        if (lat in dataset["latitude"].values) and (lon in dataset["longitude"].values):
            # Extract t2m values for the specific latitude and longitude across all times and steps
            t2m_values = dataset[variable].sel(latitude=lat, longitude=lon).values.ravel()
            # GFS at this resolution is 3-hourly, F000 and F003 are the latest forecasts until
            # a new model run begins. At the end you want to include only F000 to be able to interpolate
            # (given you are not forecasting the full range, 14 days because then you will not have
            # F000 ... Fxxx at the end)
            # So [:-1] is there to not include F003 here.
            data[(lat, lon)] = t2m_values[:-1]

    df = pd.DataFrame(data, index=index)
    return df


def interpolate_to_resolution(df, resolution):
    df = df.resample(resolution).asfreq()
    df = df.interpolate(method="linear")
    # Drop the last value again (which was only there to make interpolation easier)
    return df.iloc[:-1]


def get_X(points, start_date, end_date):
    dataset = get_gfs_data(start_date, end_date)
    df_X_3_hourly = extract_gfs_data(
        points=points,
        dataset=dataset,
        variable="t2m",
        start_date=start_date,
        end_date=end_date,
    )
    df_X = interpolate_to_resolution(df_X_3_hourly, resolution="15T")
    return df_X


def get_y(start_date, end_date):
    client = EntsoePandasClient(api_key=os.getenv("ENTSOE_KEY"))
    series = client.query_load(
        country_code="NL",
        start=pd.Timestamp(start_date, tz="UTC"),
        end=pd.Timestamp(end_date, tz="UTC"),
    )
    series.index = series.index.tz_convert("UTC").tz_localize(None)
    return series
