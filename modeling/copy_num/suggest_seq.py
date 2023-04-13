import numpy as np
import pandas as pd
from scipy.optimize import minimize
from modeling.copy_num.consts import *
from modeling.copy_num.main import combine_all_features, get_current_best_model


PROMOTER_MOD_IDX = [2, 3, 4, 5, 24, 25, 26, 27, 35]
PROMOTER_FIXED = [i for i in range(36) if i not in PROMOTER_MOD_IDX]


def promoter_restrictions(seq):
    for i, v in enumerate(seq):
        if i in PROMOTER_FIXED and v != RNAp_seq_original[i]:
            return False
    return True


# Define the objective function
def diff(seq, target_copy_num, model, restriction_funcs):
    # Check if the sequence violates any of the restrictions
    if not all([rs(seq) for rs in restriction_funcs]):
        return np.inf

    # Compute the difference between the predicted and target copy number
    data_df = pd.DataFrame({"Promoter Sequence": seq, "Copy Number": 0})
    X, y = combine_all_features(data_df, x_col="Promoter Sequence", y_col='Copy Number',
                                **{"promotor_strength": None,
                                   "pssm_score": None,
                                   "motifs_pval": None,
                                   })

    pred_copy_num = model.predict(X.reshape(1, -1))[0]
    return abs(pred_copy_num - target_copy_num)


def suggest_promotor_for_copy_num(target_copy_num=100):
    # starting point for the optimization
    start_seq = RNAp_seq_original
    model = get_current_best_model()
    restriction_funcs = [promoter_restrictions]

    result = minimize(diff, start_seq, args=(target_copy_num, model, restriction_funcs))
    optimal_seq = result.x
    print(optimal_seq)


if __name__ == '__main__':
    suggest_promotor_for_copy_num(target_copy_num=100)
