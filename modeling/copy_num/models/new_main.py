from modeling.copy_num.data_prep.pre_process import get_features_df
from modeling.copy_num.models.Parameters_Tuning.best_param_to_xl import get_best_params_set_xgb, \
    find_optimal_alpha_Lasso
from modeling.copy_num.models.models_functions import model

if __name__ == '__main__':
    data = get_features_df()
    RNAp_X = data['RNAp_X']
    RNAp_y = data['RNAp_y']
    RNAi_X = data['RNAi_X']
    RNAi_y = data['RNAi_y']
    X_shared_model = data['X_shared']
    Y_shared_model = data['Y_shared']

    # RNAp
    Best_param_p_xgb = get_best_params_set_xgb(RNAp_X, RNAp_y, "xgb_RNAp")
    model(RNAp_X, RNAp_y, "xgboost", "pRNA", Best_param_p_xgb, save_plots=True)

    Best_alpha_p = find_optimal_alpha_Lasso(RNAp_X, RNAp_y, "lasso_RNAp")
    model(RNAp_X, RNAp_y, model_name="lasso", data_name="pRNA", Best_param=Best_alpha_p, save_plots=True)

    # RNAi
    Best_param_i_xgb = get_best_params_set_xgb(RNAi_X, RNAi_y, "xgb_RNAi")
    model(RNAi_X, RNAi_y, "xgboost", "iRNA", Best_param_i_xgb, save_plots=True)

    Best_alpha_i = find_optimal_alpha_Lasso(RNAi_X, RNAi_y, "lasso_RNAi")
    model(RNAi_X, RNAi_y, model_name="lasso", data_name="iRNA", Best_param=Best_alpha_i, save_plots=True)

    # shared model
    Best_param_shared_xgb = get_best_params_set_xgb(X_shared_model, Y_shared_model, "xgb_RNA_shared")
    model(X_shared_model, Y_shared_model, "xgboost", "shared model", Best_param_shared_xgb, save_plots=True)

    Best_alpha_shared = find_optimal_alpha_Lasso(X_shared_model, Y_shared_model, "lasso_shared")
    model(X_shared_model, Y_shared_model, model_name="lasso", data_name="shared model", Best_param=Best_alpha_shared, save_plots=True)
