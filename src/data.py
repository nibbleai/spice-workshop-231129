import pandas as pd
from sklearn.model_selection import train_test_split as _train_test_split

from src.config import config
from src.directories import directories
from src.schemas import TaxiColumn, WeatherColumn

__all__ = [
    'get_train_dataset', 'get_weather_data', 'train_test_split', 'get_target'
]

TRAIN_TAXI_DATA_FILENAME = 'nyc-taxi-2015.csv'
WEATHER_DATA_FILENAME = 'nyc-weather-2015.csv'


def get_train_dataset():
    return pd.read_csv(
        directories.taxi_data_dir / TRAIN_TAXI_DATA_FILENAME,
        parse_dates=[TaxiColumn.PICKUP_TIME, TaxiColumn.DROPOFF_TIME]
    )


def get_weather_data():
    return pd.read_csv(
        directories.weather_data_dir / WEATHER_DATA_FILENAME,
        parse_dates=[WeatherColumn.DATE]
    ).assign(**{WeatherColumn.DATE: lambda x: x[WeatherColumn.DATE].dt.date})


def train_test_split(data):
    return _train_test_split(
        data.sort_values(TaxiColumn.PICKUP_TIME),
        test_size=config.test_size,
        shuffle=False
    )


def get_target(data):
    """Target is the total duration of trip, in seconds"""
    target = (
        data[TaxiColumn.DROPOFF_TIME] - data[TaxiColumn.PICKUP_TIME]
    ).dt.seconds
    return target.rename(config.target)
