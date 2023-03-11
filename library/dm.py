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


def split_by_difficulty_numerical(df: pd.DataFrame):
    cp_df = normalize_numerical(df.copy())

    cp_df[DIFFICULTY_COLUMN] = Difficulty.UNKNOWN

    for column in cp_df[cp_df.columns.difference([DIFFICULTY_COLUMN])]:
        q_series = pd.qcut(
            cp_df[column],
            q=[0, 0.10, 0.25, 0.75, 0.90, 1],
            labels=[
                f"{Difficulty.HARD}-first",
                f"{Difficulty.MEDIUM}-first",
                Difficulty.EASY,
                f"{Difficulty.MEDIUM}-last",
                f"{Difficulty.HARD}-last",
            ])

        q_series = q_series.map({  # type: ignore
            f"{Difficulty.MEDIUM}-first": Difficulty.MEDIUM,
            f"{Difficulty.MEDIUM}-last": Difficulty.MEDIUM,
            f"{Difficulty.HARD}-first": Difficulty.HARD,
            f"{Difficulty.HARD}-last": Difficulty.HARD,
        })

        cp_unknown_index = cp_df.loc[cp_df[DIFFICULTY_COLUMN]
                                     == Difficulty.UNKNOWN].index
        cp_df.iloc[cp_unknown_index, cp_df.columns.get_loc(
            DIFFICULTY_COLUMN)] = q_series.iloc[cp_unknown_index]  # type: ignore

        cp_nan_index = cp_df.loc[cp_df[DIFFICULTY_COLUMN].isnull()].index
        cp_df.iloc[cp_nan_index, cp_df.columns.get_loc(
            DIFFICULTY_COLUMN)] = Difficulty.UNKNOWN

        def merge_conflicts(a: Difficulty, b: Difficulty, c: Difficulty):
            a_or_b_index = cp_df.loc[
                (cp_df[DIFFICULTY_COLUMN] == a) |
                (cp_df[DIFFICULTY_COLUMN] == b)
            ].index

            # type: ignore
            c_index = q_series.iloc[a_or_b_index].loc[q_series == c].index

            cp_df.iloc[c_index, cp_df.columns.get_loc(
                DIFFICULTY_COLUMN)] = Difficulty.CONFLICT

        merge_conflicts(Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)
        merge_conflicts(Difficulty.HARD, Difficulty.MEDIUM, Difficulty.EASY)
        merge_conflicts(Difficulty.HARD, Difficulty.EASY, Difficulty.MEDIUM)

    return cp_df
