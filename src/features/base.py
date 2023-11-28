from src.schemas import TaxiColumn
from src.features.registry import registry

for col_name, feature_name in [
    (TaxiColumn.PICKUP_LON, "pickup_lon"),
    (TaxiColumn.PICKUP_LAT, "pickup_lat"),
    (TaxiColumn.DROPOFF_LON, "dropoff_lon"),
    (TaxiColumn.DROPOFF_LAT, "dropoff_lat"),
    (TaxiColumn.PICKUP_TIME, "pickup_time"),
]:
    registry.register(
        lambda data, col_name=col_name: data.loc[:, col_name],
        name=feature_name
    )
