"""This modules defines schemas of datasets used in the demo."""

from spice.utils import StrEnum

__all__ = ['TaxiColumn', 'WeatherColumn']


class TaxiColumn(StrEnum):
    """Column names in the taxi dataset"""

    VENDOR_ID = 'vendor_id'
    PICKUP_TIME = 'pickup_datetime'
    DROPOFF_TIME = 'dropoff_datetime'
    PASSENGER_COUNT = 'passenger_count'
    TRIP_DISTANCE = 'trip_distance'
    PICKUP_LON = 'pickup_longitude'
    PICKUP_LAT = 'pickup_latitude'
    RATE_CODE = 'rate_code'
    STORE_AND_FWD_FLAG = 'store_and_fwd_flag'
    DROPOFF_LON = 'dropoff_longitude'
    DROPOFF_LAT = 'dropoff_latitude'
    PAYMENT_TYPE = 'payment_type'
    FARE_AMOUNT = 'fare_amount'
    EXTRA = 'extra'
    MTA_TAX = 'mta_tax'
    TIP_AMOUNT = 'tip_amount'
    TOLLS_AMOUNT = 'tolls_amount'
    IMP_SURCHARGE = 'imp_surcharge'
    TOTAL_AMOUNT = 'total_amount'


class WeatherColumn(StrEnum):
    """Column names in the weather dataset"""

    DATE = 'DATE'
    PRECIPITATIONS = 'PRCP'
    MAX_TEMPERATURE = 'TMAX'
    MIN_TEMPERATURE = 'TMIN'
    AVERAGE_WIND_SPEED = 'AWND'
