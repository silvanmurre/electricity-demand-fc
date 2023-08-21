import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from entsoe import EntsoePandasClient
from herbie import FastHerbie
from shapely.geometry import Point


# TODO: Fully customizable for different countries and maps (resolutions)
def get_raster_points_inside_nl(plot=False, raster_size=0.25):
    shpfilename = shpreader.natural_earth(
            resolution='110m',
            category='cultural',
            name='admin_0_countries')
    countries = gpd.read_file(shpfilename)
    netherlands = countries[countries.NAME == 'Netherlands']
    
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
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        ax.add_geometries(netherlands.geometry, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none')
        grid_gdf.plot(ax=ax, markersize=10, color='red', transform=ccrs.PlateCarree())
        inside_points.plot(ax=ax, markersize=10, color='blue', transform=ccrs.PlateCarree())
        
        ax.set_extent([minx, maxx, miny, maxy], ccrs.PlateCarree())
        ax.coastlines(resolution='110m')
        plt.show()
    
    return inside_points


def get_gfs_data(start_date, end_date):
    # We don't need the forecast from end_date + 6 hours
    end_date = end_date - timedelta(hours=6)
    
    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="6H"
    )
    
    # TODO: Stop redownloading already downloaded and saved products.
    H = FastHerbie(DATES=dates, fxx=range(0, 6, 3), model="gfs", product="pgrb2.0p50", save_dir="data", max_threads=16, verbose=True)
    ds = H.xarray(":TMP:2 m above ground")
    return ds


def extract_gfs_data(points: gpd.GeoDataFrame, dataset, variable, start_date: datetime, end_date: datetime):
    index = pd.date_range(
        start=start_date,
        end=end_date,
        freq="3H",
        inclusive='left'
    )
    
    data  = {}
    for idx, row in points.iterrows():
        point = row.geometry
        lat, lon = point.y, point.x
        
        # Check if latitude and longitude are within the bounds of the xarray Dataset
        if (lat in dataset['latitude'].values) and (lon in dataset['longitude'].values):
            # Extract t2m values for the specific latitude and longitude across all times and steps
            t2m_values = dataset['t2m'].sel(latitude=lat, longitude=lon).values.ravel()
            data[(lat, lon)] = t2m_values
    
    df = pd.DataFrame(data, index=index)  
    return df


def get_demand_entsoe(start_date, end_date):
    load_dotenv()
    client = EntsoePandasClient(api_key=os.getenv("ENTSOE_KEY"))
    series = client.query_load(
        country_code="NL", 
        start=pd.Timestamp(start_date, tz='UTC'), 
        end=pd.Timestamp(end_date, tz='UTC'), 
    )
    return series
