from src.data import get_weather_data

from spice.utils import StrEnum

__all__ = ['Resource', 'get_resources']


class Resource(StrEnum):
    WEATHER = 'weather'


def get_resources():
    return {Resource.WEATHER: get_weather_data()}
