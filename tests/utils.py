import os
import pathlib

import pandas as pd

dir_path: pathlib.Path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))


def read_dataset(
    problem_type: str, dataset_name: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    base_path: pathlib.Path = dir_path / "datasets" / problem_type / dataset_name
    df_train: pd.DataFrame = pd.read_csv(base_path / "train.csv")
    df_test: pd.DataFrame = pd.read_csv(base_path / "test.csv")
    label_column: str = "survival_status" if problem_type == "survival" else "class"
    X_train, y_train = df_train.drop(label_column, axis=1), df_train[label_column]
    X_test, y_test = df_test.drop(label_column, axis=1), df_test[label_column]

    # survival status columns should contains string values: '1' and '0'
    if problem_type == 'survival':
        y_train = y_train.astype(int).astype(str)
        y_test = y_test.astype(int).astype(str)

    return X_train, y_train, X_test, y_test
