import pandas as pd
from pandas.testing import assert_frame_equal

from recette.steps import prep_step_dummy, prep_step_other


def test_prep_step_other():
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
            "animal": ["cat", "cat", "cat", "cat", "cat", None],
            "size": [11, None, 13, 17, 19, 23],
        }
    ).astype(dtypes)

    step_other = prep_step_other(
        train_df, cols=["color", "animal"], threshold=0.5, other_cat="abcd"
    )
    output_df = step_other(test_df)

    expected_df = pd.DataFrame(
        {
            "color": pd.Categorical(
                ["red", "red", "abcd", "abcd", "abcd", "abcd"],
                categories=["red", "abcd"],
            ),
            "animal": ["cat", "cat", "cat", "cat", "cat", None],
            "size": [11, None, 13, 17, 19, 23],
        }
    ).astype(dtypes)
    assert_frame_equal(output_df, expected_df)


def test_prep_step_dummy():
    dtypes = {"color": "category", "animal": "category", "size": "UInt8"}
    train_df = pd.DataFrame(
        {
            "color": pd.Categorical(
                ["red", "red", None, "blue", "blue", "blue"], categories=["red", "blue"]
            ),
            "animal": pd.Categorical(
                ["cat", "cat", "cat", "dog", "dog", "rat"],
                categories=["rat", "cat", "dog"],
            ),
            "size": [None, 1, 2, 3, 5, 7],
        }
    ).astype(dtypes)
    test_df = pd.DataFrame(
        {
            "color": ["red", "red", "yellow", "blue", "blue", "green"],
            "animal": ["cat", None, "cat", "cat", "cat", "dog"],
            "size": [11, None, 13, 17, 19, 23],
        }
    ).astype(dtypes)

    step_dummy = prep_step_dummy(train_df, cols=["color", "animal"], sep="_")
    output_df = step_dummy(test_df)

    expected_df = pd.DataFrame(
        {
            "size": [11, None, 13, 17, 19, 23],
            "color_blue": [0, 0, 0, 1, 1, 0],
            "animal_cat": [1, None, 1, 1, 1, 0],
            "animal_dog": [0, None, 0, 0, 0, 1],
        },
        dtype="UInt8",
    )
    assert_frame_equal(output_df, expected_df)
