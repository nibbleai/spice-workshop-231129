__all__ = ['config']


class _Config:
    def __init__(self):
        self.test_size = .33
        self.target = 'trip_duration'

        self.features = {
            "hours_bins": 4,
            # Ideal number of clusters has been determined
            # using the elbow method on the taxi dataset
            "coordinates_clusters": 4,
        }


config = _Config()
