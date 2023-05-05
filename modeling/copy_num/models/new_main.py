from modeling.copy_num.data_prep.pre_process import get_features_df
from modeling.copy_num.models.models_functions import model

from modeling.copy_num.features.pssm_feature import is_high_copy_number
from modeling.copy_num.models.models_functions import prepare_model_data

if __name__ == '__main__':
    data = get_features_df()
    RNAp_X = data['RNAp_X']
    RNAp_y = data['RNAp_y']
    RNAi_X = data['RNAi_X']
    RNAi_y = data['RNAi_y']
    X_shared_model = data['X_shared']
    Y_shared_model = data['Y_shared']

    # run models

    # RNAp
    model(None, RNAp_X, RNAp_y, model_name="lasso", data_name="pRNA")
    model(None, RNAp_X, RNAp_y, model_name="xgboost", data_name="pRNA")

    # RNAi
    model(None, RNAi_X, RNAi_y, model_name="lasso", data_name="iRNA")
    model(None, RNAi_X, RNAi_y, model_name="xgboost", data_name="iRNA")

    # shared model
    model(None, X_shared_model, Y_shared_model, model_name="lasso", data_name="shared model")
    model(None, X_shared_model, Y_shared_model, model_name="xgboost", data_name="shared model")



