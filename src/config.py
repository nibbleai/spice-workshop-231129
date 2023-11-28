__all__ = ['config']

class _Config:
    def __init__(self):
        self.test_size = .33
        self.target = 'trip_duration'


config = _Config()
