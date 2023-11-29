import logging

from spice import Generator

from src.data import get_target, get_train_dataset, train_test_split
from src.features import registry
from src.model import evaluate, get_model
from src.preprocessing import preprocess


def main():
    data = get_train_dataset()
    target = get_target(data)
    data, target = preprocess(data, target)

    train_data, test_data = train_test_split(data)
    train_target = target.loc[train_data.index]
    test_target = target.loc[test_data.index]

    generator = Generator(registry, features=["pickup_hour", "pickup_weekday"])

    train_features = generator.fit_transform(
        train_data,
        tags={"dataset": "train"},
    ).to_pandas()
    test_features = generator.transform(
        test_data,
        tags={"dataset": "test"},
    ).to_pandas()

    logging.info("Training model...")
    model = get_model().fit(train_features, train_target)

    metrics = evaluate(model, features=test_features, target=test_target)
    logging.info(f"Model metrics are:\n{metrics}")


if __name__ == "__main__":
    main()
