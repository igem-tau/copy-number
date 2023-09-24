import pandas as pd
from pathlib import Path
import pickle


def get_current_file_parent_path(file) -> Path:
    return Path(file).parent.resolve()


def is_feature_selected(feature: str, selected_features: Optional[List[str]]) -> bool:
    return selected_features is None or feature in selected_features


def get_selected_features_path(rna_type: str, model: str = '') -> Path:
    model = f'{model}_' if model else model
    return Path(get_current_file_parent_path(__file__), '..', 'data',
                f'{model}RNA{rna_type}_selected_features.pkl')


def write_selected_features(features: list, rna_type: str, model: str = ''):
    selected_features_path = get_selected_features_path(rna_type, model)

    with open(selected_features_path, 'wb') as f:
        pickle.dump(features, f)


# TODO - update the selected features usage - call once and pass trough to the features generation functions
def get_selected_features(rna_type, model: str = '') -> set:
    selected_features_path = get_selected_features_path(rna_type, model)

    if not selected_features_path.exists():
        raise FileNotFoundError(
            'get_selected_features: you must first run the process that saves the selected features in to a file'
        )

    with open(selected_features_path, 'rb') as f:
        features = pickle.load(f)

    return set(features)


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
