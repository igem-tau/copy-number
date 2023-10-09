import pandas as pd
from joblib import dump, load
import numpy as np
from minepy import MINE
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn import metrics
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import davies_bouldin_score
import sklearn


def PCA_density_clustering(X, y, eps, min_samples, PCA_flag=0, plot_flag=0):
    if PCA_flag == 1:
        # Perform PCA for dimensional reduction
        n_components = 2  # Set the number of components you want to reduce to (2 for plotting purposes)
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X)
        db = DBSCAN(eps=eps, min_samples=int(min_samples)).fit(X)
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        if plot_flag == 1:
            clustered_df = pd.DataFrame(X)
            clustered_df['Copy Number'] = y
            clustered_df['Cluster'] = labels

            grouped_features = clustered_df.groupby(by='Cluster')
            grouped_features_mean = grouped_features['Copy Number'].mean()
            grouped_features_std = grouped_features['Copy Number'].std()
            legend_array_mean=grouped_features_mean.loc[0:9]
            legend_array_mean[10]=grouped_features_mean.loc[-1]
            legend_array_mean = [round(num, 2) for num in legend_array_mean]

            legend_array_std = grouped_features_std.loc[0:9]
            legend_array_std[10] = grouped_features_std.loc[-1]
            legend_array_std = [round(num, 2) for num in legend_array_std]

            legend_array = np.transpose(np.array([legend_array_mean,legend_array_std]))

            # Create a 1x2 subplot layout for two plots side by side
            fig, axs = plt.subplots(1, 2, figsize=(16, 6))

            # Plot the PCA results in the first subplot
            axs[0].scatter(X[:, 0], X[:, 1], cmap='viridis')
            axs[0].set_xlabel('Principal Component 1')
            axs[0].set_ylabel('Principal Component 2')
            axs[0].set_title('PCA for Dimensional Reduction of Feature Matrix X')
            print("Estimated number of clusters: %d" % n_clusters_)
            print("Estimated number of noise points:%d" % n_noise_)
            print("Estimated number of clustered points:%d" % (RNAp_X.shape[1] - n_noise_))

            # Plot the clustered points in the second subplot
            unique_labels = set(labels)
            core_samples_mask = np.zeros_like(labels, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True

            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]
                    # continue

                class_member_mask = labels == k

                xy = X[class_member_mask & core_samples_mask]
                axs[1].plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=14,label='_nolegend_')

                xy = X[class_member_mask & ~core_samples_mask]
                axs[1].plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=6)

            axs[1].legend(legend_array, loc='upper right')
            axs[1].set_xlabel('Principal Component 1')
            axs[1].set_ylabel('Principal Component 2')
            axs[1].set_title(f"Estimated number of clusters: {n_clusters_}")

            plt.show()
        else:
            grouped_features = 'tzlil hapermut'


    else:

        # Perform DBSCAN clustering on the reduced data
        db = DBSCAN(eps=eps, min_samples=int(min_samples)).fit(X)
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        # print("Estimated number of clusters: %d" % n_clusters_)
        # print("Estimated number of noise points:%d" % n_noise_)
        # print("Estimated number of clustered points:%d" % (RNAp_X.shape[1] - n_noise_))
        if plot_flag == 1:
            clustered_df = pd.DataFrame(X)
            clustered_df['Copy Number'] = y
            clustered_df['Cluster'] = labels

            grouped_features = clustered_df.groupby(by='Cluster')
            # Plot the clustered points in the second subplot
            unique_labels = set(labels)
            core_samples_mask = np.zeros_like(labels, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True

            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]
                    continue

                class_member_mask = labels == k

                xy = X[class_member_mask & core_samples_mask]
                plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k",
                         markersize=14)

                xy = X[class_member_mask & ~core_samples_mask]
                plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k",
                         markersize=6)

            plt.title(f"Estimated number of clusters: {n_clusters_}")

            plt.show()
        else:
            grouped_features = 'tzlil hapermut'

    # Create a DataFrame with the original features and their corresponding cluster labels

    # if plot_flag == 1:
    #     clustered_df = pd.DataFrame(X)
    #     clustered_df['Copy Number'] = y
    #     clustered_df['Cluster'] = labels
    #
    #     grouped_features = clustered_df.groupby(by='Cluster')
    # else:
    #     grouped_features = 'tzlil hapermut'

    try:
        score = sklearn.metrics.davies_bouldin_score(X, labels)
    except:
        score = np.inf
    return (grouped_features, score)


def density_clustering_parameter_tuning(X, y, distances, min_sampless, PCA_flag):
    best_score = np.inf
    for eps in distances:
        for min_samples in min_sampless:
            (g, score) = PCA_density_clustering(X, y, eps, min_samples, PCA_flag, 0)
            if score < best_score:
                best_score = score
                best_eps = eps
                best_min_samples = min_samples
    return (best_eps, best_min_samples, best_score)


def permutate_features_df(X):
    permutated_X = X.copy()
    # Iterate through each column and shuffle its values
    for column in permutated_X.columns:
        # Get the values of the current column
        values = permutated_X[column].values

        # Shuffle the values randomly
        np.random.shuffle(values)

        # Update the column with shuffled values
        permutated_X[column] = values
    T_perm_X=permutated_X.T
    for column in T_perm_X.columns:
        # Get the values of the current column
        values = T_perm_X[column].values

        # Shuffle the values randomly
        np.random.shuffle(values)

        # Update the column with shuffled values
        T_perm_X[column] = values


    return np.transpose(T_perm_X)


def calculate_pv(X, y, eps, min_samples, PCA_flag, score, perm_num):
    counter = 0
    for i in range(perm_num):
        permu_x = permutate_features_df(X)
        (g, score_permu) = PCA_density_clustering(permu_x, y, eps, min_samples, PCA_flag, 0)
        if score_permu < score:
            counter = counter + 1
    return counter / perm_num


data = load('DataFrames_with_features_and_correlations.joblib')
RNAp_X = data['RNAp_X']
RNAp_y = data['RNAp_y']

# Assuming X is a 2-dimensional array (matrix)
# X = np.transpose(RNAp_X)
X = RNAp_X
y = RNAp_y
permutated_X = permutate_features_df(X)


eps=1
min_samples=13
PCA_flag = 1
plot_flag = 0

# eps=0.8
# min_samples=5
# PCA_flag = 1
# plot_flag = 0


(grouped_features,score)=PCA_density_clustering(X,RNAp_y,eps,min_samples,PCA_flag,1)
grouped_features_mean=grouped_features['Copy Number'].mean()
grouped_features_std=grouped_features['Copy Number'].std()

a=5



# distances = np.linspace(0.7, 1.5, 20)
# min_sampless = np.linspace(10, 16, 7)
# (best_eps,best_min_samples,score)=density_clustering_parameter_tuning(X,y,distances,min_sampless,PCA_flag)
# perm_num=100
# pv=calculate_pv(X,y,best_eps,best_min_samples,PCA_flag,score,perm_num)
# print(f'pv is {pv}')
# print(f'the best eps is {best_eps}')
# print(f' the best min_samples is {best_min_samples}')
# print(f'the score is {score}')
#
#
# (grouped_features,score)=PCA_density_clustering(X,RNAp_y,best_eps,best_min_samples,1,1)


# (grouped_features,score)=PCA_density_clustering(X,RNAp_y,1,13,1,1)
# pv=calculate_pv(X,y,1,13,PCA_flag,score,perm_num)
# print(f'pv is {pv}')
# print(f'the best eps is {best_eps}')
# print(f' the best min_samples is {best_min_samples}')
# print(f'the score is {score}')

