from typing import Callable

import pandas as pd
from pandas._typing import Dtype

StepType = Callable[[pd.DataFrame], pd.DataFrame]
PrepType = Callable[[pd.DataFrame], StepType]
