import pandas as pd
from pandas.testing import assert_frame_equal

from recette.steps import prep_step_dummy, prep_step_other
from recette.utils import combine


def test_combine():
    dtypes = {"color": "category", "animal": "category", "size": "UInt8"}
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

    prep = combine(
        prep_step_other(cols=["color", "animal"], threshold=0.5, other_cat="abcd"),
        prep_step_dummy(cols=["color", "animal"], sep="_"),
    )
    step = prep(train_df)
    output_df = step(test_df)

    expected_df = pd.DataFrame(
        {
            "size": [11, None, 13, 17, 19, 23],
            "color_abcd": [0, 0, 1, 1, 1, 1],
            "animal_dog": [0, 0, 0, 0, 0, None],
        },
        dtype="UInt8",
    )
    assert_frame_equal(output_df, expected_df)
