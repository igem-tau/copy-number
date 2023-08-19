import os
from pathlib import Path, PosixPath
import pickle


DIR = os.path.dirname(os.path.abspath(__file__))
SELECTED_FEATURES_FILE_NAME = os.path.join(DIR, "selected_features.pkl")


def get_current_file_parent_path(file) -> PosixPath:
    return Path(file).parent.resolve()


def write_selected_features(features: list):
    with open(SELECTED_FEATURES_FILE_NAME, 'wb') as f:
        pickle.dump(features, f)


def get_selected_features() -> set:
    with open(SELECTED_FEATURES_FILE_NAME, 'rb') as f:
        features = pickle.load(f)

    return set(features)


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