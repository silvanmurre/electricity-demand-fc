import geopandas as gpd
import numpy as np
import pandas as pd

from datetime import datetime
from shapely.geometry import Point

from electricity_demand_fc.data_retrieval import (
    get_raster_points_inside_nl,
    get_gfs_data,
    extract_gfs_data,
    get_demand_entsoe,
)


def test_get_raster_points_inside_nl():
    expected_points_list = [
        Point(4.00000, 51.50000),
        Point(4.50000, 51.50000),
        Point(4.50000, 52.00000),
        Point(4.50000, 52.50000),
        Point(5.00000, 51.50000),
        Point(5.00000, 52.00000),
        Point(5.00000, 52.50000),
        Point(5.00000, 53.00000),
        Point(5.50000, 51.50000),
        Point(5.50000, 52.00000),
        Point(5.50000, 52.50000),
        Point(5.50000, 53.00000),
        Point(6.00000, 51.00000),
        Point(6.00000, 51.50000),
        Point(6.00000, 52.00000),
        Point(6.00000, 52.50000),
        Point(6.00000, 53.00000),
        Point(6.50000, 52.00000),
        Point(6.50000, 52.50000),
        Point(6.50000, 53.00000),
        Point(7.00000, 53.00000),
    ]
    expected_points_gseries = gpd.GeoSeries(expected_points_list)
    expected_points_gdf = gpd.GeoDataFrame(geometry=expected_points_gseries)

    actual_points = get_raster_points_inside_nl(raster_size=0.5)

    assert expected_points_gdf.equals(actual_points), "Expected points do not match actual points"


def test_extract_gfs_data():
    latitudes = [
        51.5,
        51.5,
        52.0,
        52.5,
        51.5,
        52.0,
        52.5,
        53.0,
        51.5,
        52.0,
        52.5,
        53.0,
        51.0,
        51.5,
        52.0,
        52.5,
        53.0,
        52.0,
        52.5,
        53.0,
        53.0,
    ]
    longitudes = [
        4.0,
        4.5,
        4.5,
        4.5,
        5.0,
        5.0,
        5.0,
        5.0,
        5.5,
        5.5,
        5.5,
        5.5,
        6.0,
        6.0,
        6.0,
        6.0,
        6.0,
        6.5,
        6.5,
        6.5,
        7.0,
    ]
    index = pd.MultiIndex.from_tuples(list(zip(latitudes, longitudes)))
    timestamps = [
        "2023-01-01 00:00:00",
        "2023-01-01 03:00:00",
        "2023-01-01 06:00:00",
        "2023-01-01 09:00:00",
        "2023-01-01 12:00:00",
    ]
    values = [
        [
            287.25806,
            287.93805,
            287.63806,
            284.16806,
            287.92804,
            287.50806,
            286.85803,
            284.32803,
            288.21805,
            287.29803,
            285.99805,
            285.43805,
            287.63806,
            287.56805,
            287.36804,
            287.49805,
            286.84805,
            287.66806,
            287.50806,
            287.42804,
            287.42804,
        ],
        [
            285.40085,
            286.23087,
            285.53085,
            283.31085,
            286.58087,
            285.86087,
            285.14087,
            283.57086,
            287.21085,
            285.68085,
            284.60086,
            283.98087,
            286.67087,
            286.95087,
            285.18085,
            285.99084,
            284.62085,
            286.53085,
            285.91086,
            284.68085,
            285.02087,
        ],
        [
            285.55203,
            286.16202,
            285.94202,
            282.95203,
            286.23203,
            285.96204,
            284.98203,
            283.09204,
            286.79202,
            286.05203,
            284.28204,
            283.89203,
            286.58203,
            286.44202,
            286.10202,
            285.84204,
            284.66202,
            286.27203,
            286.07202,
            285.25204,
            285.47205,
        ],
        [
            286.31387,
            286.86386,
            286.61386,
            283.65387,
            287.36386,
            286.39386,
            286.09387,
            283.57385,
            288.01385,
            286.84387,
            285.37387,
            284.45386,
            287.05386,
            287.69388,
            287.33386,
            286.45386,
            286.12387,
            287.57385,
            286.94388,
            286.54385,
            286.92386,
        ],
        [
            285.19098,
            285.44098,
            285.25098,
            283.03098,
            285.41098,
            284.90097,
            284.63098,
            283.18097,
            286.16098,
            284.82098,
            283.90097,
            283.52097,
            285.91098,
            286.16098,
            285.391,
            285.171,
            284.19098,
            285.59097,
            285.40097,
            284.731,
            285.25098,
        ],
    ]
    expected_df = pd.DataFrame(values, columns=index, index=timestamps)

    points = get_raster_points_inside_nl(plot=False, raster_size=0.5)
    start_date = datetime(year=2023, month=1, day=1)
    end_date = datetime(year=2023, month=1, day=1, hour=12)
    dataset = get_gfs_data(start_date, end_date)
    actual_df = extract_gfs_data(
        points=points,
        dataset=dataset,
        variable="t2m",
        start_date=start_date,
        end_date=end_date,
    )

    assert np.allclose(expected_df, actual_df, atol=1e-5), "Dataframes are not approximately equal"


def test_get_demand_entsoe():
    expected_values = [
        [9847.0],
        [9801.0],
        [9746.0],
        [9680.0],
        [9640.0],
        [9613.0],
        [9524.0],
        [9439.0],
        [9398.0],
        [9325.0],
        [9260.0],
        [9202.0],
        [9172.0],
        [9145.0],
        [9136.0],
        [9090.0],
        [9090.0],
        [9070.0],
        [9074.0],
        [9087.0],
        [9147.0],
        [9153.0],
        [9176.0],
        [9229.0],
        [9340.0],
        [9379.0],
        [9439.0],
        [9513.0],
        [9619.0],
        [9716.0],
        [9775.0],
        [9768.0],
        [9908.0],
        [10017.0],
        [10105.0],
        [10169.0],
        [10192.0],
        [10131.0],
        [10043.0],
        [9970.0],
        [10036.0],
        [10093.0],
        [10025.0],
        [10085.0],
        [10271.0],
        [10355.0],
        [10451.0],
        [10494.0],
    ]

    start_date = datetime(year=2023, month=1, day=1)
    end_date = datetime(year=2023, month=1, day=1, hour=12)
    df = get_demand_entsoe(start_date, end_date)
    actual_values = df.values.tolist()

    assert expected_values == actual_values, f"Expected {expected_values} but got {actual_values}"
