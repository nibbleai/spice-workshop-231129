from pathlib import Path
from types import SimpleNamespace

__all__ = ['directories']

directories = SimpleNamespace()

directories.root_dir = Path(__file__).resolve().parents[1]
directories.data_dir = directories.root_dir / 'data'
directories.taxi_data_dir = directories.data_dir / 'nyc-taxi'
directories.weather_data_dir = directories.data_dir / 'nyc-weather-res'
directories.artifacts_dir = directories.root_dir / 'artifacts'
