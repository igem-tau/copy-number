from modeling.copy_num.data_prep.post_process import get_features_df


data = get_features_df()

RNAp_X = data['RNAp_X']
RNAp_y = data['RNAp_y']
RNAi_X = data['RNAi_X']
RNAi_y = data['RNAi_y']

