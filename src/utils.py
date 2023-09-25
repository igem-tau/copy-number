import pandas as pd
from pathlib import Path
from typing import List, Optional


def get_current_file_parent_path(file) -> Path:
    return Path(file).parent.resolve()


def is_feature_selected(feature: str, selected_features: 'Optional[List[str]]') -> bool:
    return selected_features is None or feature in selected_features


def get_current_date() -> str:
    return str(pd.to_datetime("today")).split()[0]


if __name__ == '__main__':
    # print(f'the current file parent path is: {get_current_file_parent_path(__file__)}')
    # test_features = ["TTT__TC_count", "A_count", "gc_skew", "z_curve_y",
    #                  "ada", "fhlA", "Fis_26-48", "UxuR_14-2", "pssm_score",
    #                  "Predicted Promoter Strength (KbT)",
    #                  "TCMCTCCTTT", "CGCGTTWG", "WNGCNCTYYT",
    #                  "(-11, -8) predicted strength",
    #                  'G_-35', 'T_-30', 'A_-19', 'C_-2'
    #                  ]
    # write_selected_features(test_features)
    pass
