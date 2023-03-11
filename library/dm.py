import pandas as pd
import numpy as np

from enum import Enum


DIFFICULTY_COLUMN = 'difficulty'


class Difficulty(Enum):
    HARD = 'hard'
    MEDIUM = 'medium'
    EASY = 'easy'
    UNKNOWN = 'unknown'
    CONFLICT = 'conflict'


class BinaryValue(Enum):
    ZERO = 0
    ONE = 1


def split_by_dtype(df: pd.DataFrame):
    binary = pd.DataFrame()
    categorical = pd.DataFrame()
    numerical = pd.DataFrame()

    for column in df:
        if df[column].dtypes == "float64" or df[column].dtypes == "int64":
            numerical[column] = df[column]
        if df[column].dtypes == "object":
            categorical[column] = df[column]
        if df[column].dtypes == "boolean":
            binary[column] = df[column]

    return binary, categorical, numerical


# Binary diffuculty decision maker unit

def decision_boundary_binary(df: pd.DataFrame,
                             binary_df: pd.DataFrame):
    binary_rate = len(df) / len(binary_df)
    return Difficulty.HARD if (binary_rate < .3 or binary_rate > .7) else Difficulty.EASY


def split_by_difficulty_binary(df: pd.DataFrame):
    cp_df = df.copy()

    cp_df[DIFFICULTY_COLUMN] = Difficulty.UNKNOWN

    for column in cp_df[cp_df.columns.difference([DIFFICULTY_COLUMN])]:
        zeros_df = cp_df.loc[cp_df[column] == BinaryValue.ZERO]
        cp_df[DIFFICULTY_COLUMN] = cp_df[DIFFICULTY_COLUMN].iloc[zeros_df.index] = decision_boundary_binary(
            cp_df, zeros_df)
        ones_df = cp_df.loc[cp_df[column] == BinaryValue.ONE]
        cp_df[DIFFICULTY_COLUMN] = cp_df[DIFFICULTY_COLUMN].iloc[ones_df.index] = decision_boundary_binary(
            cp_df, ones_df)

    return cp_df

# Numerical diffuculty decision maker unit


def normalize_numerical(df: pd.DataFrame):
    cp_df = df.copy()

    for column in cp_df:
        cp_df[column] = np.log1p(cp_df[column])

    return cp_df


def split_by_quantiles(df: pd.DataFrame,
                       column: str):
    q_series = pd.Series(data=pd.Categorical(values=[Difficulty.UNKNOWN.value] * len(df),
                                             categories=[
                                                 Difficulty.EASY.value,
                                                 Difficulty.MEDIUM.value,
                                                 Difficulty.HARD.value,
                                                 Difficulty.UNKNOWN.value,
                                                 Difficulty.CONFLICT.value
    ])
    )

    q_percentage = [0, .16, .33, .66, .84, 1]
    q_categories = [
        Difficulty.HARD.value,
        Difficulty.MEDIUM.value,
        Difficulty.EASY.value,
        Difficulty.MEDIUM.value,
        Difficulty.HARD.value,
    ]

    q_values = [df[column].quantile(percentage) for percentage in q_percentage]

    q_pairs = []
    for i in range(1, len(q_values)):
        q_pairs.append((q_values[i - 1], q_values[i]))

    for q_category, (q_from, q_to) in zip(q_categories, q_pairs):
        q_index = df.loc[
            (df[column] >= q_from) &
            (df[column] <= q_to)
        ].index
        q_series.iloc[q_index] = q_category

    return q_series


def split_by_difficulty_numerical(df: pd.DataFrame):
    cp_df = normalize_numerical(df.copy())

    cp_df[DIFFICULTY_COLUMN] = Difficulty.UNKNOWN.value

    for column in cp_df[cp_df.columns.difference([DIFFICULTY_COLUMN])]:
        q_series = split_by_quantiles(cp_df, column)

        cp_unknown_index = cp_df.loc[cp_df[DIFFICULTY_COLUMN]
                                     == Difficulty.UNKNOWN.value].index
        cp_df.iloc[cp_unknown_index, cp_df.columns.get_loc(
            DIFFICULTY_COLUMN)] = q_series.iloc[cp_unknown_index]

        cp_nan_index = cp_df.loc[cp_df[DIFFICULTY_COLUMN].isnull()].index
        cp_df.iloc[cp_nan_index, cp_df.columns.get_loc(
            DIFFICULTY_COLUMN)] = Difficulty.UNKNOWN.value

        def merge_conflicts(a: Difficulty, b: Difficulty, c: Difficulty):
            ab_index = cp_df.loc[
                (cp_df[DIFFICULTY_COLUMN] == a.value) |
                (cp_df[DIFFICULTY_COLUMN] == b.value)
            ].index

            q_ab_series = q_series.iloc[ab_index]
            q_abc_series = q_ab_series.loc[q_ab_series == c.value]

            cp_df.iloc[q_abc_series.index, cp_df.columns.get_loc(
                DIFFICULTY_COLUMN)] = Difficulty.CONFLICT.value

        merge_conflicts(Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)
        merge_conflicts(Difficulty.HARD, Difficulty.MEDIUM, Difficulty.EASY)
        merge_conflicts(Difficulty.HARD, Difficulty.EASY, Difficulty.MEDIUM)

    return cp_df
