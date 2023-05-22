from modeling.copy_num.data_prep.pre_process import get_features_df
from modeling.copy_num.models.models_functions import model
from modeling.copy_num.models.models_functions import prepare_model_data
from modeling.copy_num.models.xgboost_model import converge_randomsearch
from modeling.copy_num.models.Parameters_Tuning.best_param_to_xl import get_best_params_set


if __name__ == '__main__':
    data = get_features_df()
    RNAp_X = data['RNAp_X']
    RNAp_y = data['RNAp_y']
    RNAi_X = data['RNAi_X']
    RNAi_y = data['RNAi_y']
    X_shared_model = data['X_shared']
    Y_shared_model = data['Y_shared']

    params_status="" # "active i" "active shared"
    if not len(params_status)==0:
        if params_status=="active p":
            X_train, X_test, y_train, y_test = prepare_model_data(RNAp_X, RNAp_y)
            dataset_name = "RNAp"
        elif params_status=="active i":
            X_train, X_test, y_train, y_test = prepare_model_data(RNAi_X, RNAi_y)
            dataset_name = "RNAi"
        elif params_status == "active shared":
            X_train, X_test, y_train, y_test = prepare_model_data(X_shared_model,Y_shared_model)
            dataset_name="RNA_shared"

        for i in range(5):
            [ii, kk] = converge_randomsearch(X_train, X_test, y_train, y_test,dataset_name,num_of_steps=7, nun_iter=7)

    else:
        # run models

        # RNAp
        Best_param_p=get_best_params_set("xgb_RNAp")
        # model(None, RNAp_X, RNAp_y, model_name="lasso", data_name="pRNA")
        data_name = "pRNA"
        model_name = "xgboost"
        model(RNAp_X, RNAp_y, model_name, data_name,Best_param_p)

        # RNAi
        Best_param_i=get_best_params_set("xgb_RNAi")
        data_name = "iRNA"
        model_name = "xgboost"
        # model(None, RNAi_X, RNAi_y, model_name="lasso", data_name="iRNA")
        model(RNAi_X, RNAi_y, model_name, data_name,Best_param_i)

        # shared model
        Best_param_shared=get_best_params_set("xgb_RNA_shared")
        data_name = "shared model"
        model_name = "xgboost"
        # model(None, X_shared_model, Y_shared_model, model_name="lasso", data_name="shared model")
        model(X_shared_model, Y_shared_model, model_name, data_name,Best_param_shared)
