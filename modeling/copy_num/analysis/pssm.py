import numpy as np
import pandas as pd
from Bio import motifs
from typing import Tuple


# def low_high_cp(df: pd.DataFrame) -> Tuple[pd.DataFrame]:
#     n = int(df.shape[0] * 0.2)
#     high_cp = df.nlargest(n, 'Copy Number')['Promoter Sequence (-35 to +1)']
#     low_cp = df.nsmallest(n, 'Copy Number')['Promoter Sequence (-35 to +1)']
#     return high_cp, low_cp





# TODO: add funcs that plot pssm
def get_threshold():
    pass