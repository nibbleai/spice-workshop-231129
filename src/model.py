from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error

__all__ = ['get_model', 'evaluate']


def get_model():
    return RandomForestRegressor()


def evaluate(model, *, features, target):
    predicted = model.predict(features)
    return {'mean_squared_log_error': mean_squared_log_error(target, predicted)}
