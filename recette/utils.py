from typing import Union

import pandas as pd

from recette.types import PrepType, StepType


def combine(*step_or_preps: Union[StepType, PrepType]) -> PrepType:
    """Combine a sequence of steps and preparations into a single preparation.

    Args:
        step_or_preps: Sequence of steps or preparations.

    Returns:
        Preparation applying each given steps or preparation in sequence.

    """

    def prep(df: pd.DataFrame) -> StepType:
        steps = []
        df_or_step = df
        for step_or_prep in iter(step_or_preps):
            # Apply step or prep to DataFrame, returning a DataFrame or a step
            df_or_step = step_or_prep(df)
            if isinstance(df_or_step, pd.DataFrame):
                # Store step in case it's a step
                steps.append(step_or_prep)
                df = df_or_step
            else:
                # Else, apply step to DataFrame and store step
                steps.append(df_or_step)
                df = df_or_step(df)

        def step(df: pd.DataFrame) -> pd.DataFrame:
            # Apply each step in sequence
            for step in steps:
                df = step(df)

            return df

        return step

    return prep
