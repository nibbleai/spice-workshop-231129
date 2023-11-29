from spice import Registry

registry = Registry(name="taxi_nyc")


@registry.register
def pickup_time(data):
    return data["pickup_datetime"]


@registry.register(depends=["pickup_time"])
def pickup_hour(pickup_time):
    return pickup_time.dt.hour


@registry.register(depends=["pickup_time"])
def pickup_weekday(pickup_time):
    return pickup_time.dt.weekday
