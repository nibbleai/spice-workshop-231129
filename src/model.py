import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, TimeSeriesSplit

__all__ = ['get_model', 'evaluate']

N_SPLITS = 5
SCORING = 'neg_mean_squared_log_error'


def get_model():
    return RandomForestRegressor()


def evaluate(model, *, features, target):
    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    cv_losses = cross_validate(
        model, features, target, scoring=SCORING, cv=splitter,
    )

    return (
        pd.DataFrame(cv_losses)
        .rename(columns={'test_score': SCORING})
        .agg(['mean', 'std'])
    )
