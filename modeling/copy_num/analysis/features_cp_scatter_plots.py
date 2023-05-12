from modeling.copy_num.data_prep.pre_process import get_features_df
import matplotlib.pyplot as plt

if __name__=='__main__':
    data = get_features_df()
    RNAp_X = data['RNAp_X']
    RNAp_y = data['RNAp_y']
    RNAi_X = data['RNAi_X']
    RNAi_y = data['RNAi_y']
    X_shared_model = data['X_shared']
    Y_shared_model = data['Y_shared']

    # motifs vs cp:
    fig, ax = plt.subplots(1, 2, figsize=(8, 5))
    scatter_df_p_x = RNAp_X[RNAp_X['rpoD17'] != 1]
    scatter_df_p_y = RNAp_y[RNAp_X['rpoD17'] != 1]

    scatter_df_i_x = RNAi_X[RNAi_X['rpoD18'] != 1]
    scatter_df_i_y = RNAi_y[RNAi_X['rpoD18'] != 1]

    ax[0].scatter(scatter_df_p_x['rpoD17'] * 1e3, scatter_df_p_y)
    ax[1].scatter(scatter_df_i_x['rpoD18'] * 1e3, scatter_df_i_y)

    ax[0].set_title('RNAp - rpoD17 vs. copy number')
    ax[1].set_title('RNAi - rpoD18 vs. copy number')

    ax[0].set_xlabel('pv [10^-3]')
    ax[1].set_xlabel('pv [10^-3]')
    ax[0].set_ylabel('copy number')
    plt.show()

    # AA+TT count vs. cp
    fig, ax = plt.subplots(1, 2, figsize=(8, 5))
    ax[0].scatter(RNAp_X['AA_count'], RNAp_y)
    ax[1].scatter(RNAi_X['TT_count'], RNAi_y)

    ax[0].set_title('RNAp - AA_count vs. copy number')
    ax[1].set_title('RNAi - TT_count vs. copy number')

    ax[0].set_xlabel('AA_count')
    ax[1].set_xlabel('TT_count')
    ax[0].set_ylabel('copy number')
    plt.show()

    fig, ax = plt.subplots(2, 2, figsize=(7, 7))

    ax[0, 0].hist(RNAp_y[RNAp_X['AA_count'] == 0])
    ax[0, 1].hist(RNAp_y[RNAp_X['AA_count'] == 1])
    ax[1, 0].hist(RNAp_y[RNAp_X['AA_count'] == 2])
    ax[1, 1].hist(RNAp_y[RNAp_X['AA_count'] == 3])

    ax[0, 0].set_title('AA count = 0')
    ax[0, 1].set_title('AA count = 1')
    ax[1, 0].set_title('AA count = 2')
    ax[1, 1].set_title('AA count = 3')

    fig.supylabel("copy number")
    plt.suptitle('RNAp - AA count vs. copy number')

    fig, ax = plt.subplots(1, 2, figsize=(5, 4))

    ax[0].hist(RNAi_y[RNAi_X['TT_count'] == 1])
    ax[1].hist(RNAi_y[RNAi_X['TT_count'] == 2])

    ax[0].set_title('TT count = 1')
    ax[1].set_title('TT count = 2')

    fig.supylabel("copy number")
    plt.suptitle('RNAi - TT count vs. copy number')

    # pssm score vs. cp
    fig, ax = plt.subplots(1, 2, figsize=(8, 5))
    ax[0].scatter(RNAp_X['pssm_score'], RNAp_y)
    ax[1].scatter(RNAi_X['pssm_score'], RNAi_y)

    ax[0].set_title('RNAp - pssm score vs. copy number')
    ax[1].set_title('RNAi - pssm score vs. copy number')

    # ax[0].set_xlabel('pv [10^-3]')
    # ax[1].set_xlabel('pv [10^-3]')
    ax[0].set_ylabel('copy number')
    fig.supxlabel("pssm score")
    plt.show()