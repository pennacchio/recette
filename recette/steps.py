from typing import Callable, List

import pandas as pd


StepType = Callable[[pd.DataFrame], pd.DataFrame]


def step_other_learner(
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
            if other_cat not in df.loc[:, col].cat.categories:
                df.loc[:, col] = df.loc[:, col].cat.add_categories(other_cat)
            df.loc[~df.loc[:, col].isin(above_threshold_cats[col]), col] = other_cat
            df.loc[:, col] = df.loc[:, col].cat.remove_unused_categories()

        return df

    return step_other
