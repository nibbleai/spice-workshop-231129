from src.schemas import TaxiColumn
from src.invariants import SECONDS_IN_HOUR

__all__ = ['preprocess']


def preprocess(trips, duration):
    coordinate_indices = _coordinate_anomalies_indices(trips)
    duration_indices = _duration_anomalie_indices(duration)

    indices = set([*coordinate_indices, *duration_indices])
    return trips.drop(indices), duration.drop(indices)


def _coordinate_anomalies_indices(trips):
    """Return indices of trips with incoherent coordinates"""
    return trips[
        (trips[TaxiColumn.PICKUP_LON] == 0) |
        (trips[TaxiColumn.PICKUP_LAT] == 0) |
        (trips[TaxiColumn.DROPOFF_LON] == 0) |
        (trips[TaxiColumn.DROPOFF_LAT] == 0)
    ].index


def _duration_anomalie_indices(duration):
    """Return indices of trips with incoherent duration"""
    return duration[(duration > SECONDS_IN_HOUR) | (duration <= 0)].index
