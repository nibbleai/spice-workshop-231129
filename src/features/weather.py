import pandas as pd

from src.features.registry import registry
from src.resources import Resource
from src.schemas import WeatherColumn

from . import logger


@registry.register(
    name="[artifact]_weather",
    depends=["pickup_date"],
    resources=[Resource.WEATHER],
)
def weather(pickup_date, weather_resource):
    """Weather features based on pickup date."""
    logger.info(
        "Weather resource ranges from "
        f"{weather_resource[WeatherColumn.DATE].min()} to "
        f"{weather_resource[WeatherColumn.DATE].max()}"
    )

    right_on = [WeatherColumn.DATE]
    return pd.DataFrame(pickup_date).merge(
        weather_resource,
        left_on=pickup_date.name,
        right_on=right_on,
        how='left'
    ).set_index(pickup_date.index)


@registry.register(name="is_raining", depends=["[artifact]_weather"])
def is_raining(weather):
    """Is it raining on the day of pickup"""
    return (weather[WeatherColumn.PRECIPITATIONS] > 0).astype(int)
