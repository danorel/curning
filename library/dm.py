import pandas as pd
import numpy as np

from enum import Enum


DIFFICULTY_COLUMN = 'difficulty'


class Difficulty(Enum):
    HARD = 'hard'
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

def meassure_difficulty_binary(df: pd.DataFrame, column: str):
    b_values = [BinaryValue.ZERO.value, BinaryValue.ONE.value]

    b_pairs = []
    for b_value in b_values:
        b_rate = len(df.loc[df[column] == b_value]) / len(df)
        if (b_rate < .3):
            b_pairs.append(Difficulty.HARD.value)
        else:
            b_pairs.append(Difficulty.EASY.value)

    return b_pairs


def split_by_binary(df: pd.DataFrame,
                    column: str):

    b_series = pd.Series(data=pd.Categorical(values=[Difficulty.UNKNOWN.value] * len(df),
                                             categories=[
                                                 Difficulty.EASY.value,
                                                 Difficulty.HARD.value,
                                                 Difficulty.UNKNOWN.value,
                                                 Difficulty.CONFLICT.value
    ])
    )

    b_pairs = meassure_difficulty_binary(df, column)

    for (b_difficulty, b_value) in b_pairs:
        c_index = df.loc[df[column] == b_value].index
        b_series.iloc[c_index] = b_difficulty

    return b_series


def split_by_difficulty_binary(df: pd.DataFrame):
    cp_df = df.copy()

    cp_df[DIFFICULTY_COLUMN] = Difficulty.UNKNOWN.value

    for column in cp_df[cp_df.columns.difference([DIFFICULTY_COLUMN])]:
        b_series = split_by_binary(cp_df, column)

        cp_unknown_index = cp_df.loc[cp_df[DIFFICULTY_COLUMN]
                                     == Difficulty.UNKNOWN.value].index
        cp_df.iloc[cp_unknown_index, cp_df.columns.get_loc(
            DIFFICULTY_COLUMN)] = b_series.iloc[cp_unknown_index]

        cp_nan_index = cp_df.loc[cp_df[DIFFICULTY_COLUMN].isnull()].index
        cp_df.iloc[cp_nan_index, cp_df.columns.get_loc(
            DIFFICULTY_COLUMN)] = Difficulty.UNKNOWN.value

        def merge_conflicts(a: Difficulty, b: Difficulty):
            a_index = cp_df.loc[cp_df[DIFFICULTY_COLUMN] == a.value].index

            b_a_series = b_series.iloc[a_index]
            b_ab_series = b_a_series.loc[b_a_series == b.value]

            cp_df.iloc[b_ab_series.index, cp_df.columns.get_loc(
                DIFFICULTY_COLUMN)] = Difficulty.CONFLICT.value

        merge_conflicts(Difficulty.EASY, Difficulty.HARD)
        merge_conflicts(Difficulty.HARD, Difficulty.EASY)

    return cp_df

# Numerical diffuculty decision maker unit


def normalize_numerical(df: pd.DataFrame):
    cp_df = df.copy()

    for column in cp_df:
        cp_df[column] = np.log1p(cp_df[column])

    return cp_df


def meassure_difficulty_numerical(df: pd.DataFrame, column: str):
    q_percentage = [0, .2, .8, 1]
    q_values = [df[column].quantile(percentage) for percentage in q_percentage]

    q_pairs = []
    for i in range(1, len(q_values)):
        q_pairs.append((q_values[i - 1], q_values[i]))

    return q_pairs


def split_by_quantiles(df: pd.DataFrame,
                       column: str):
    q_series = pd.Series(data=pd.Categorical(values=[Difficulty.UNKNOWN.value] * len(df),
                                             categories=[
                                                 Difficulty.EASY.value,
                                                 Difficulty.HARD.value,
                                                 Difficulty.UNKNOWN.value,
                                                 Difficulty.CONFLICT.value
    ])
    )

    q_difficulties = [
        Difficulty.HARD.value,
        Difficulty.EASY.value,
        Difficulty.HARD.value,
    ]

    q_pairs = meassure_difficulty_numerical(df, column)

    for q_difficulty, (q_from, q_to) in zip(q_difficulties, q_pairs):
        q_index = df.loc[
            (df[column] >= q_from) &
            (df[column] <= q_to)
        ].index
        q_series.iloc[q_index] = q_difficulty

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

        def merge_conflicts(a: Difficulty, b: Difficulty):
            a_index = cp_df.loc[cp_df[DIFFICULTY_COLUMN] == a.value].index

            b_a_series = q_series.iloc[a_index]
            b_ab_series = b_a_series.loc[b_a_series == b.value]

            cp_df.iloc[b_ab_series.index, cp_df.columns.get_loc(
                DIFFICULTY_COLUMN)] = Difficulty.CONFLICT.value

        merge_conflicts(Difficulty.HARD, Difficulty.EASY)
        merge_conflicts(Difficulty.EASY, Difficulty.HARD)

    return cp_df

# Categorical diffuculty decision maker unit


def meassure_difficulty_categorical(df: pd.DataFrame, column: str):
    c_values = df[column].unique()

    c_pairs = []
    for c_value in c_values:
        c_rate = len(df.loc[df[column] == c_value]) / len(df)
        if (c_rate < .2):
            c_pairs.append((Difficulty.HARD.value, c_value))
        else:
            c_pairs.append((Difficulty.EASY.value, c_value))

    return c_pairs


def split_by_categoricals(df: pd.DataFrame,
                          column: str):

    c_series = pd.Series(data=pd.Categorical(values=[Difficulty.UNKNOWN.value] * len(df),
                                             categories=[
                                                 Difficulty.EASY.value,
                                                 Difficulty.HARD.value,
                                                 Difficulty.UNKNOWN.value,
                                                 Difficulty.CONFLICT.value
    ])
    )

    c_pairs = meassure_difficulty_categorical(df, column)

    for (c_difficulty, c_value) in c_pairs:
        c_index = df.loc[df[column] == c_value].index
        c_series.iloc[c_index] = c_difficulty

    return c_series


def split_by_difficulty_categorical(df: pd.DataFrame):
    cp_df = df.copy()

    cp_df[DIFFICULTY_COLUMN] = Difficulty.UNKNOWN.value

    for column in cp_df[cp_df.columns.difference([DIFFICULTY_COLUMN])]:
        c_series = split_by_categoricals(cp_df, column)

        cp_unknown_index = cp_df.loc[cp_df[DIFFICULTY_COLUMN]
                                     == Difficulty.UNKNOWN.value].index
        cp_df.iloc[cp_unknown_index, cp_df.columns.get_loc(
            DIFFICULTY_COLUMN)] = c_series.iloc[cp_unknown_index]

        cp_nan_index = cp_df.loc[cp_df[DIFFICULTY_COLUMN].isnull()].index
        cp_df.iloc[cp_nan_index, cp_df.columns.get_loc(
            DIFFICULTY_COLUMN)] = Difficulty.UNKNOWN.value

        def merge_conflicts(a: Difficulty, b: Difficulty):
            a_index = cp_df.loc[cp_df[DIFFICULTY_COLUMN] == a.value].index

            b_a_series = c_series.iloc[a_index]
            b_ab_series = b_a_series.loc[b_a_series == b.value]

            cp_df.iloc[b_ab_series.index, cp_df.columns.get_loc(
                DIFFICULTY_COLUMN)] = Difficulty.CONFLICT.value

        merge_conflicts(Difficulty.HARD, Difficulty.EASY)
        merge_conflicts(Difficulty.EASY, Difficulty.HARD)

    return cp_df
