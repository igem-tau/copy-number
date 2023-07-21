import matplotlib.pyplot as plt
from src.data_prep.pre_process import get_features_df

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

    fig, ax = plt.subplots()
    ax.hist(RNAp_y[RNAp_X['AA_count'] == 0], label='0', histtype='step', density=True)  # Plot histogram of nums1
    ax.hist(RNAp_y[RNAp_X['AA_count'] == 1], label='1', histtype='step', density=True)
    ax.hist(RNAp_y[RNAp_X['AA_count'] == 2], label='2', histtype='step', density=True)  # Plot histogram of nums1
    ax.hist(RNAp_y[RNAp_X['AA_count'] == 3], label='3', histtype='step', density=True)
    plt.legend()
    fig.supylabel('copy number')
    plt.suptitle('RNAp - AA count vs. copy number')
    # plt.show()

    fig, ax = plt.subplots()
    ax.hist(RNAi_y[RNAi_X['TT_count'] == 1], label='1', histtype='step', density=True)  # Plot histogram of nums1
    ax.hist(RNAi_y[RNAi_X['TT_count'] == 2], label='2', histtype='step', density=True)
    plt.legend()
    fig.supylabel('copy number')
    plt.suptitle('RNAi - TT count vs. copy number')
    plt.show()

    # # pssm score vs. cp
    fig, ax = plt.subplots(1, 2, figsize=(8, 5))
    ax[0].scatter(RNAp_X['pssm_score'], RNAp_y)
    ax[1].scatter(RNAi_X['pssm_score'], RNAi_y)

    ax[0].set_title('RNAp - pssm score vs. copy number')
    ax[1].set_title('RNAi - pssm score vs. copy number')

    # ax[0].set_xlabel('pv [10^-3]')
    # ax[1].set_xlabel('pv [10^-3]')
    ax[0].set_ylabel('copy number')
    fig.supxlabel('pssm score')
    plt.show()