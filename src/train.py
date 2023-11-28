import logging

from spice import Generator

from src.data import get_train_dataset, get_target, train_test_split
from src.features.registry import registry
from src.features import base, spatial, temporal, weather
from src.io_ import save_generator, save_model
from src.preprocessing import preprocess
from src.model import get_model, evaluate
from src.resources import get_resources


def main():
    data = get_train_dataset()
    target = get_target(data)
    data, target = preprocess(data, target)

    train_data, test_data = train_test_split(data)
    train_target = target.loc[train_data.index]
    test_target = target.loc[test_data.index]

    feature_generator = Generator(
        registry,
        resources=get_resources(),
        features=[
            "cyclical_pickup_hour",
            "quantile_bin_hour",
            "is_raining",
            "is_rush_hour",
            "euclidean_distance",
            "manhattan_distance",
            "euclidean_manhattan_ratio",
            "pca_pickup",
            "pca_dropoff",
            "pickup_cluster",
            "dropoff_cluster"
        ]
    )
    train_features = (
        feature_generator
        .fit_transform(train_data, tags={'dataset': 'train'})
        .to_pandas()
    )

    test_features = (
        feature_generator
        .transform(test_data, tags={'dataset': 'test'})
        .to_pandas()
    )

    logging.info("Training model...")
    model = get_model().fit(train_features, train_target)

    metrics = evaluate(model, features=test_features, target=test_target)
    logging.info(f"Model metrics are:\n{metrics}")

    logging.info("Saving generator...")
    save_generator(feature_generator)

    logging.info("Saving model...")
    save_model(model)


if __name__ == '__main__':
    main()
