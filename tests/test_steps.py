import pandas as pd
from pandas.testing import assert_frame_equal

from recette.steps import step_other_learner


def test_step_other_learner():
    # Input data. Note that categorical columns are used
    dtypes = {"color": "category", "animal": "category", "size": "Int64"}
    train_df = pd.DataFrame(
        {
            "color": ["red", "red", "red", "blue", "blue", "green"],
            "animal": ["cat", "cat", "cat", "dog", "dog", "dog"],
            "size": [None, 1, 2, 3, 5, 7],
        }
    ).astype(dtypes)
    test_df = pd.DataFrame(
        {
            "color": ["red", "red", "yellow", "blue", "blue", "green"],
            "animal": ["cat", "cat", "cat", "cat", "cat", "cat"],
            "size": [11, None, 13, 17, 19, 23],
        }
    ).astype(dtypes)

    # Compute output
    step_other = step_other_learner(train_df, cols=["color", "animal"], threshold=0.5)
    output_df = step_other(test_df)

    # Check if it's right
    expected_df = pd.DataFrame(
        {
            "color": pd.Categorical(
                ["red", "red", "other", "other", "other", "other"],
                categories=["red", "other"],
            ),
            "animal": ["cat", "cat", "cat", "cat", "cat", "cat"],
            "size": [11, None, 13, 17, 19, 23],
        }
    ).astype(dtypes)
    assert_frame_equal(output_df, expected_df)
