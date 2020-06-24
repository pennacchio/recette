from typing import List
from uuid import uuid4

import pandas as pd
from toolz import curry

from recette.types import Dtype, StepType


@curry
def prep_step_other(
    df: pd.DataFrame,
    cols: List[str],
    threshold: float = 0.05,
    other_cat: str = "other",
    ignore_na: bool = False,
) -> StepType:
    """Collapse categories with proportion below threshold.

    Args:
        df: Pandas DataFrame to process.
        cols: Columns to process.
        threshold: Proportion threshold below which categories are collapsed.
        other_cat: Category used for collapsed categories.
        ignore_na: Ignore NAs when computing proportions.

    Returns:
        Step that collapses all categories other than the ones above proportion
        threshold.

    """
    # Get categories above proportion threshold for each column
    above_threshold_cats = {
        col: df.loc[:, col]
        .value_counts(normalize=True, dropna=ignore_na)
        .loc[lambda srs: srs >= threshold]
        .index
        for col in cols
    }

    def step_other(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in cols:
            # Add other category if it isn't already present
            if other_cat not in df.loc[:, col].cat.categories:
                df.loc[:, col] = df.loc[:, col].cat.add_categories(other_cat)

            # Find out which rows should be collapsed
            is_other_cat = ~(
                df.loc[:, col].isin(above_threshold_cats[col]) | df.loc[:, col].isna()
            )

            # Collapse categories and remove them
            df.loc[is_other_cat, col] = other_cat
            df.loc[:, col] = df.loc[:, col].cat.remove_unused_categories()

        return df

    return step_other


@curry
def prep_step_dummy(
    df: pd.DataFrame,
    cols: List[str],
    drop_first: bool = True,
    keep_cols: bool = False,
    sep: str = "=",
    dtype: Dtype = "UInt8",
) -> StepType:
    """Convert categorical columns to dummy columns.

    Args:
        df: Pandas DataFrame to process.
        cols: Columns to process.
        drop_first: For k categories, create k - 1 dummy variables instead of k
            by removing the first category.
        keep_cols: Keep processed columns (in addition to dummy columns).
        sep: Separator between column name and category for dummy column names.
        dtype: Dummies data type. Note that if NAs are expected, a nullable data
            type should be used.

    Returns:
        Step that converts categorical columns to dummy columns.

    """
    # Store known categories for each column, dropping the first if appropriate
    cols_cats = [df.loc[:, col].cat.categories[int(drop_first) :] for col in cols]

    def step_dummy(df: pd.DataFrame) -> pd.DataFrame:
        original_df, df = df, df.copy()

        # Convert all unknown categories to temporary ones
        tmp_cats = []
        for col, cats in zip(cols, cols_cats):
            # Get unknown categories
            unknown_cats = df.loc[:, col].cat.categories.difference(cats)

            # Make temporary category
            tmp_cat = str(uuid4())  # Ensure no collapse with any category
            tmp_cats.append(tmp_cat)
            df.loc[:, col] = df.loc[:, col].cat.add_categories(tmp_cat)

            # Convert unknown categories to it
            df.loc[df.loc[:, col].isin(unknown_cats), col] = tmp_cat

            # Remove unknown categories and set the remaining categories order
            df.loc[:, col] = (
                df.loc[:, col]
                .cat.remove_categories(unknown_cats)
                .cat.set_categories(cats.insert(0, tmp_cat))
            )

        dummies_df = pd.get_dummies(df.loc[:, cols], columns=cols, prefix_sep=sep)

        # Remove temporary category dummies
        tmp_dummy_cols = dummies_df.columns.intersection(
            [col + sep + str(tmp_cat) for col, tmp_cat in zip(cols, tmp_cats)]
        )
        dummies_df = dummies_df.drop(columns=tmp_dummy_cols)

        # Propagate NAs
        for col, cats in zip(cols, cols_cats):
            dummy_cols = [col + sep + str(cat) for cat in cats]
            dummies_df.loc[df.loc[:, col].isna(), dummy_cols] = pd.NA

        # Cast dummies to given dtype
        dummies_df = dummies_df.astype(dtype)

        return original_df.drop(columns=[] if keep_cols else cols).join(dummies_df)

    return step_dummy
