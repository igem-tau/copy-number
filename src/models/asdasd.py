from src.data_prep.pre_process import get_RNAp_data, split_for_testing,equal_bins_data, create_fasta_file

TARGET_COLUMN = 'Copy Number'


RNAp_data = get_RNAp_data()
RNAp_data, RNAp_stratify_col = equal_bins_data(RNAp_data)
RNAp_X = RNAp_data.drop(TARGET_COLUMN, axis=1)
RNAp_y = RNAp_data[TARGET_COLUMN]

RNAp_data_train_val, RNAp_data_test = split_for_testing(RNAp_X, RNAp_y, stratify_by=RNAp_stratify_col)
RNAp_stratify_train_val, RNAp_stratify_test = split_for_testing(RNAp_stratify_col, RNAp_y,
                                                                            stratify_by=RNAp_stratify_col)

a=5