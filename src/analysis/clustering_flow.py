import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sklearn
from src.data_prep.pre_process import generate_features

def PCA_density_clustering(X,y, eps, min_samples, PCA_flag, plot_flag):
    if PCA_flag:
        # Perform PCA for dimensional reduction
        n_components = 2  # Set the number of components you want to reduce to (2 for plotting purposes)
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X)
        db = DBSCAN(eps=eps, min_samples=int(min_samples)).fit(X)
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        if plot_flag:
            clustered_df = pd.DataFrame(X)
            clustered_df['labels'] = y
            clustered_df['Cluster'] = labels
            grouped_features = clustered_df.groupby(by='Cluster')
            fig, axs = plt.subplots(1, 2, figsize=(16, 6))

            # Plot the PCA results in the first subplot
            label_color_map = {label: color for label, color in
                               zip(clustered_df['labels'].unique(), plt.cm.tab20.colors)}

            for label, color in label_color_map.items():
                label_indices = clustered_df[clustered_df['labels'] == label].index
                axs[0].scatter(X[label_indices, 0], X[label_indices, 1], color=color, label=label)

            axs[0].set_xlabel('Principal Component 1')
            axs[0].set_ylabel('Principal Component 2')
            axs[0].set_title('PCA for Dimensional Reduction of Feature Matrix X')
            axs[0].legend()


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

            # axs[1].legend(legend_array, loc='upper right')
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
        if plot_flag:
            clustered_df = pd.DataFrame(X)
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
    try:
        score = sklearn.metrics.davies_bouldin_score(X, labels)
    except:
        score = np.inf
    return (grouped_features, score)

def density_clustering_parameter_tuning(X, distances, min_sampless, PCA_flag,PLOT_FLAG):
    best_score = np.inf
    for eps in distances:
        for min_samples in min_sampless:
            (g, score) = PCA_density_clustering(X, eps, min_samples, PCA_flag, PLOT_FLAG)
            if score < best_score:
                best_score = score
                best_eps = eps
                best_min_samples = min_samples
    return (best_eps, best_min_samples, best_score)

def permutate_features_df(X):
    permutated_X = X.copy()
    # Iterate through each column and shuffle its values
    for column in permutated_X.columns:
        values = permutated_X[column].values
        np.random.shuffle(values)
        permutated_X[column] = values
    T_perm_X=permutated_X.T
    for column in T_perm_X.columns:
        values = T_perm_X[column].values
        np.random.shuffle(values)
        T_perm_X[column] = values
    return np.transpose(T_perm_X)

def calculate_pv(X, eps, min_samples, PCA_flag, score,PLOT_FLAG,perm_num):
    counter = 0
    for i in range(perm_num):
        permu_x = permutate_features_df(X)
        (g, score_permu) = PCA_density_clustering(permu_x, eps, min_samples, PCA_flag,PLOT_FLAG)
        if score_permu < score:
            counter = counter + 1
    return counter / perm_num

def data_prp(file):
    row_data = pd.read_excel(file)
    # RNAp_X = data['RNAP_promoter']
    # RNAi_X = data['RNAI_promoter']
    # y=data['genus']
    # f=generate_features(RNAp_X,rna_type ='p')
    data=5
    return data,y

def run(file,PCA_FLAG,PV_FLAG,PARAM_TUNINING_FLAG,PLOT_FLAG):
    X,y=data_prp(file)
    if PARAM_TUNINING_FLAG[0]:
        distances = np.linspace(0.7, 10, 100)
        min_sampless = np.linspace(5, 55, 100)
        best_eps, best_min_samples, best_score=density_clustering_parameter_tuning(X, distances, min_sampless, PCA_FLAG,PLOT_FLAG)
        print(f' Best epsilon parameter is : {best_eps}\nBest min samples parameter is : {min_sampless}\nThe score for best parameters is: {best_score} ')
        PARAM_TUNINING_FLAG=[False,[best_eps,best_min_samples]]
    else:
       [],best_score= PCA_density_clustering(X, y, PARAM_TUNINING_FLAG[1][0], PARAM_TUNINING_FLAG[1][1],PCA_FLAG,PLOT_FLAG)

    if PV_FLAG:
        pv=calculate_pv(X, PARAM_TUNINING_FLAG[1][0], PARAM_TUNINING_FLAG[1][1], PCA_FLAG, best_score, PLOT_FLAG,permu_num=1000)
        print(f' PV for parameters best_eps:{best_eps},best_min_samples:{best_min_samples} is {pv} ')






if __name__=='__main__':
    file=r''
    PCA_FLAG=True
    PV_FLAG=True
    PLOT_FLAG=True
    PARAM_TUNINING_FLAG=[True,None]   # [True,None] | [False ,[eps=4,min_samples=7]]
    run(file,PCA_FLAG,PV_FLAG,PARAM_TUNINING_FLAG,PLOT_FLAG)