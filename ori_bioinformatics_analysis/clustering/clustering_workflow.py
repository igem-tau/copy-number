import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering, AffinityPropagation
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
import warnings
import random
from sklearn.decomposition import PCA
from dpca import *

def clustering_parameter_tuning(X,CLUSTER_TYPE_POOL, PCA_flag):
    best_score = np.inf
    best_s=np.inf
    g_dic=dict()
    final_dic=dict()
    cluster_name=None
    for CLUSTER_TYPE in CLUSTER_TYPE_POOL:
        if CLUSTER_TYPE=='HDB':
            distances = np.linspace(0.0, 0.05, 2)
            min_cluster_size=range(5,30,10)
            for c_s in min_cluster_size:
                for eps in distances:
                        parameters=[eps,c_s]
                        (labels, score) = clustering(X,CLUSTER_TYPE, parameters, PCA_flag)
                        if score < best_score:
                            best_score = score
                            best_eps = eps
                            best_c_s=c_s
            try:
                g_dic['HDB']=[{'best eps':best_eps,'best cluster_size':best_c_s}, best_score]
            except:
                print('HDB is not good enough')

        elif CLUSTER_TYPE=='KM':
            num_sump = range(2,50,25)
            for n_sumpels in num_sump:
                    parameters = [n_sumpels]
                    (labels, score) = clustering(X,CLUSTER_TYPE, parameters, PCA_flag)
                    if score < best_score:
                        best_score = score
                        best_n_sumpels = n_sumpels
            try:
                g_dic['KM']=[{'best n_sumpels':best_n_sumpels}, best_score]
            except:
                print('KM is not good enough')

        elif CLUSTER_TYPE=='SP':
            n_clusters = range(2, 100,50)
            affinity = ['rbf', 'cosine']
            n_neighbors = range(5, 50,20)
            eigen_solver = ['arpack', 'lobpcg']
            for n_c in n_clusters:
                for af in affinity:
                    for n_n in n_neighbors:
                        for e_s in eigen_solver:
                            parameters = [n_c,af,n_n,e_s]
                            (labels, score) = clustering(X,CLUSTER_TYPE, parameters, PCA_flag)
                            if score < best_score:
                                best_score = score
                                best_n_c ,best_af,best_n_n,best_e_s= n_c,af,n_n,e_s
            try:
                g_dic['SP']=[{'best n_clusters':best_n_c,'best affinity':best_af,'best n_neighbors':best_n_n,'best eigen_solver':best_e_s}, best_score]
            except:
                print('SP is not good enough')
                continue
        elif CLUSTER_TYPE == 'DPCA':
            n_clusters, dis_meth=np.linspace(2, 70, 2), ["Gaussion",'manhattan_distance']
            for k in n_clusters:
                for meth in dis_meth:
                    parameters = [k,meth]
                    (labels, score) = clustering(X, CLUSTER_TYPE, parameters, PCA_flag)
                    if score < best_score:
                        best_score = score
                        best_k, best_meth = k, meth
            try:
                g_dic['DPCA']=[{'best k': best_k, 'best meth': best_meth}, best_score]
            except:
                print('DPCA is not good enough')


        elif CLUSTER_TYPE=='GMM':
            n_components = range(2, 10,5)
            covariance_type = ['full', 'tied', 'diag', 'spherical']
            init_params = [ 'random_from_data'] #'kmeans', 'k-means++', 'random',
            max_iter = range(100, 500,250)
            tol = np.linspace(0.01, 0.1,1)
            reg_covar = np.linspace(0.1, 1,1)
            warm_start = [True, False]
            for n_component in n_components:
                for covariance_t in covariance_type:
                    for init_param in init_params:
                        for max_it in max_iter:
                            for to in tol:
                                for reg_c in reg_covar:
                                        for warm_s in warm_start:
                                            parameters = [n_component, covariance_t, init_param, max_it, to, reg_c, warm_s]
                                            (labels, score) = clustering(X, CLUSTER_TYPE, parameters, PCA_flag)
                                            if score < best_score:
                                                best_score = score
                                                best_n_component,best_covariance_type,best_init_params,best_max_iter,best_tol,best_reg_covar,best_warm_start=n_component, covariance_t, init_param, max_it, to, reg_c, warm_s
            try:
                g_dic['GMM']=[{"best_n_components":best_n_component, "best_covariance_type":best_covariance_type, "best_init_params":best_init_params, "best_max_iter":best_max_iter, "best_tol":best_tol, "best_reg_covar":best_reg_covar, "best_warm_start":best_warm_start}, best_score]
            except:
                print('GMM is not good enough')

        elif CLUSTER_TYPE=='AF':
            damping = np.linspace(0.5, 0.99,2)
            preference = range(-1, 1)
            max_iter = range(100, 500,250)
            for damp in damping:
                for pref in preference:
                    for max_ite in max_iter:
                            parameters=[ damp, pref ,max_ite]
                            (labels, score) = clustering(X, CLUSTER_TYPE, parameters, PCA_flag)
                            if score < best_score:
                                best_score = score
                                best_damp,best_pref,best_n_max_ite = damp, pref ,max_ite
            try:
                g_dic['AF']=[{"best_damp": best_damp, "best_pref": best_pref,"best_n_max_ite": best_n_max_ite}, best_score]
            except:
                print('SP is not good enough')
    for k,s in g_dic.items():
        if s[1]<best_s:
            best_s=s[1]
            final_dic=g_dic[k]
            cluster_name=k
    return final_dic,cluster_name







def permutate_features_df(X):
    permutated_X = X.copy()
    # Iterate through each column and shuffle its values
    for column in range(permutated_X.shape[0]):
        values = permutated_X.iloc[column,:]
        np.random.shuffle(values)
        permutated_X.iloc[column,:] = values
    T_perm_X=permutated_X.T
    for column in range(T_perm_X.shape[0]):
        values = permutated_X.iloc[column,:]
        np.random.shuffle(values)
        T_perm_X.iloc[column,:] = values
    return np.transpose(T_perm_X)
def rand_data(X):
    return np.random.rand(X.shape[0],X.shape[1])*np.max(np.max(X))
def calculate_pv(X,parameters,PCA_flag,score,perm_num=1):
    counter = 0
    for i in range(perm_num):
        permu_x = permutate_features_df(X)# |permutate_features_df(X)
        (g, score_permu) = clustering(permu_x,CLUSTER_TYPE,parameters,PCA_flag)
        if score_permu < score:
            counter = counter + 1
    return permu_x,counter / perm_num

def data_prp(file):
    df=pd.read_excel(file)
    y,data=df.loc[:,'family'],df.iloc[:,4:] #df.apply(calculate_max, axis=1)
    if data.ndim==1:
        data=data.values.reshape(-1,1)
    return data,df,y
def calculate_max(row):
    return max(row['cARS_RNAI_promoter'], row['cARS_RNAP_promoter'])

def clustering(X,CLUSTER_TYPE,parameters,PCA_flag):
    if X.ndim == 1:
        data = np.ndarray(X, (-1, 1))
    else:
        data = X.to_numpy()
    if CLUSTER_TYPE=='KM' and type(parameters)==list:
        kmeans = KMeans(n_clusters=parameters[0])
        kmeans.fit(data)
        labels=kmeans.labels_
    elif CLUSTER_TYPE=='HDB' and type(parameters)==list:
        eps,min_cluster_size = parameters[0], parameters[1]
        db = HDBSCAN(cluster_selection_epsilon=eps,min_cluster_size=min_cluster_size).fit(X)
        labels = db.labels_
    elif CLUSTER_TYPE=='DPCA':
        k,meth= parameters[0], parameters[1]
        dists = getDistanceMatrix(data)
        dc = select_dc(dists)
        rho = get_density(dists, dc, method=meth)  # we can use other distance such as 'manhattan_distance'
        deltas, nearest_neiber = get_deltas(dists, rho)
        # draw_decision(rho, deltas,data, name='h' + "_decision.jpg")
        centers = find_centers_K(rho, deltas, k)
        labels = cluster_PD(rho, centers, nearest_neiber)

    elif CLUSTER_TYPE=='DB' and type(parameters)==list:
        eps,min_cluster_size = parameters[0], parameters[1]
        db = DBSCAN(cluster_selection_epsilon=eps,min_cluster_size=min_cluster_size).fit(X)
        labels = db.labels_
    elif CLUSTER_TYPE=='AF' and type(parameters)==list:
        af = AffinityPropagation(damping=parameters[0], preference=parameters[1], max_iter=parameters[2]).fit(X)
        labels = af.labels_
    elif CLUSTER_TYPE=='SP' and type(parameters)==list:
        n_clusters,affinity,n_neighbors,eigen_solver = parameters[0], parameters[1] , parameters[2] ,parameters[3]
        sp = SpectralClustering(n_clusters=n_clusters, affinity=affinity, n_neighbors=n_neighbors, eigen_solver=eigen_solver).fit(X)
        labels = sp.labels_
    elif CLUSTER_TYPE=='GMM' and type(parameters)==list:
        n_components, covariance_type, init_params, max_iter, tol, reg_covar, warm_start = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6]
        gmm = GaussianMixture()
        gmm.set_params(n_components=n_components, covariance_type=covariance_type, init_params=init_params, max_iter=max_iter, tol=tol, reg_covar=reg_covar, warm_start=warm_start)
        gmm.fit(X)
        labels = gmm.predict(data)
    if len(set(labels))<3:
        score=np.inf
    else:
        score_0 = -metrics.silhouette_score(X, labels)
        if score_0 <-0.3 :
            score_1=metrics.calinski_harabasz_score(X, labels)
            score_2=sklearn.metrics.davies_bouldin_score(X, labels)
            score = np.mean([score_2,1/score_1]) #-score_0
        else:
            score=np.inf
    return labels, score


def plot_clusters(X,y,labels,score,CLUSTER_TYPE,RUN_NAME):
    if  X.ndim==1:
        X=X.values.reshape(-1,1)
    if X.shape[1]>3:
        X=perform_pca(X, 2)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    fig,ax=plt.subplots(1,2)
    clustered_df = pd.DataFrame(X)
    clustered_df['labels'] = y
    clustered_df['Cluster'] = labels
    label_color_map = {label: color for label, color in zip(clustered_df['labels'].unique(), plt.cm.tab20.colors)}

    if X.shape[1]==1:
        sorted_x_h = np.sort(X.iloc[:,0])
        sorted_y_h = np.sort(X.iloc[:,0])
        sorted_x_l = np.sort(X.iloc[:,0])[::-1]
        sorted_y_l = np.sort(X.iloc[:,0])[::-1]
    elif  X.shape[1]==2:
        sorted_x_h = np.sort(X.iloc[:,0])
        sorted_y_h = np.sort(X.iloc[:,1])
        sorted_x_l = np.sort(X.iloc[:,0])[::-1]
        sorted_y_l = np.sort(X.iloc[:,1])[::-1]

    for label, color in label_color_map.items():
        label_indices = clustered_df[clustered_df['labels'] == label].index
        if X.shape[1]==2:
            ax[0].scatter(clustered_df.loc[label_indices, X.columns[0]], clustered_df.loc[label_indices, X.columns[1]], color=color, label=label,s=2)
            ax[1].scatter(clustered_df.loc[label_indices, X.columns[0]], clustered_df.loc[label_indices, X.columns[1]], color=color, label=label,s=3)
        elif  X.shape[1]==1:
            ax[0].scatter(clustered_df.loc[label_indices, X.columns[0]], clustered_df.loc[label_indices, X.columns[0]], color=color, label=label, s=2)
            ax[1].scatter(clustered_df.loc[label_indices, X.columns[0]], clustered_df.loc[label_indices, X.columns[0]], color=color, label=label, s=2)

    ax[0].set_title('Family (color) score plot')
    ax[0].legend(fontsize='small')
    desired_percentage = 0.95
    index_threshold = int(len(sorted_x_h) * desired_percentage)
    x_threshold_h = sorted_x_h[index_threshold]
    y_threshold_h = sorted_y_h[index_threshold]
    x_threshold_l = sorted_x_l[index_threshold]
    y_threshold_l = sorted_y_l[index_threshold]
    ax[1].set_xlim(x_threshold_l,x_threshold_h)
    ax[1].set_ylim(y_threshold_l,y_threshold_h)
    ax[1].set_title('Zoom on 90% of the data')
    plt.show()
    # fig.savefig(f'{RUN_NAME} natural')
    fig2, ax2 = plt.subplots(1, 2)
    hexadecimal_alphabets = '0123456789ABCDEF'
    colors = ["#" + ''.join([random.choice(hexadecimal_alphabets) for j in range(6)]) for i in range(n_clusters_)]
    cluster_color_map = {label: color for label, color in zip(clustered_df['Cluster'].unique(), colors)}
    for label, color in cluster_color_map.items():
        label_indices = clustered_df[clustered_df['Cluster'] == label].index
        if X.shape[1]==2:
            ax2[0].scatter(clustered_df.loc[label_indices, X.columns[0]], clustered_df.loc[label_indices, X.columns[1]], color=color, label=label)
            ax2[1].scatter(clustered_df.loc[label_indices, X.columns[0]], clustered_df.loc[label_indices, X.columns[1]], color=color, label=label)
        elif X.shape[1]==1:
            ax2[0].scatter(clustered_df.loc[label_indices, X.columns[0]], clustered_df.loc[label_indices, X.columns[0]],
                           color=color, label=label)
            ax2[1].scatter(clustered_df.loc[label_indices, X.columns[0]], clustered_df.loc[label_indices, X.columns[0]],
                           color=color, label=label)

    ax2[0].set_title('Cluster (color) score plot')
    if X.shape[1]==2:
        ax2[0].set_xlabel(X.columns[0])
        ax2[0].set_ylabel(X.columns[1])
    # legend = ax2[0].legend(fontsize='small')
    # legend.set_bbox_to_anchor((0,1))
    index_threshold = int(len(sorted_x_h) * desired_percentage)
    x_threshold_h = sorted_x_h[index_threshold]
    y_threshold_h = sorted_y_h[index_threshold]
    x_threshold_l = sorted_x_l[index_threshold]
    y_threshold_l = sorted_y_l[index_threshold]
    ax2[1].set_xlim(x_threshold_l,x_threshold_h)
    ax2[1].set_ylim(y_threshold_l,y_threshold_h)



    ax2[1].set_title('Zoom on 90% of the data')
    fig2.suptitle(f'{CLUSTER_TYPE} cluster with {round(score,2)} score and {n_clusters_} clusters')
    plt.show()
    fig2.savefig(f'{RUN_NAME} clustering')

def perform_pca(X, n_components):
    if n_components not in [2, 3]:
        raise ValueError("n_components must be 2 or 3 for 2D or 3D reduction.")
    pca = PCA(n_components=n_components)
    reduced_X = pca.fit_transform(X)
    if n_components == 2:
        return pd.DataFrame(data=reduced_X[:, :2], columns=['PC1', 'PC2'])


def grouped_labels_to_excel(X, X2, y, labels, RUN_NAME):
    if  X.ndim == 1:
        X = X.values.reshape(-1, 1)
    if X.shape[1] > 3:
        X = perform_pca(X, 2)

    clustered_df = pd.DataFrame(X)
    clustered_df['labels'] = y
    clustered_df['Cluster'] = labels
    clustered_df2 = pd.DataFrame(X2)
    clustered_df2['labels'] = y
    clustered_df2['Cluster'] = labels
    grouped = clustered_df.groupby('Cluster')['labels'].value_counts().unstack().fillna(0).transpose()
    grouped2 = clustered_df2.groupby('Cluster')

    with pd.ExcelWriter(f'{RUN_NAME}.xlsx') as writer:

        for group_name, group_data in grouped2:
            sheet_name = f'Group_{group_name}'
            group_data.to_excel(writer, sheet_name=sheet_name, index=False)
        grouped.to_excel(writer, sheet_name='Summary', index=False)
        for i in range(len(grouped.columns)):
            column = grouped.columns[i]
            values = grouped[column]
            labels = grouped.index
            total_count = values.sum()
            fig = plt.figure(figsize=(6, 6))
            plt.pie(values, labels=None, autopct='%1.1f%%')
            plt.title(f'Group {column}')
            angles = plt.gca().patches
            values['num element'] = values.sum()
            for angle, label in zip(angles, labels):
                x, y = angle.center
                plt.annotate(f'{values}', (x, y), xytext=(-1.6, -1.66))

            # plt.legend(labels=labels, title='Labels', loc=(0.7, 0.65))
            # plt.show()
            # fig.savefig(f'{RUN_NAME}_group_{column}.png')


def run(file,PCA_FLAG,PV_FLAG,PARAM_TUNINING_FLAG,PLOT_FLAG,CLUSTER_TYPE,GROUPED_FLAG,RUN_NAME,CLUSTER_TYPE_POOL):
    X,X2,y=data_prp(file)
    if PARAM_TUNINING_FLAG[0]:
        par,CLUSTER_TYPE = clustering_parameter_tuning(X, CLUSTER_TYPE_POOL, PCA_FLAG)
        best_params, best_score=par[0],par[1]
        print(f'The best params are {best_params}\nThe score for best parameters is: {best_score} ')
        PARAM_TUNINING_FLAG=[False,list(best_params.values())]
    labels,score= clustering(X,CLUSTER_TYPE,PARAM_TUNINING_FLAG[1],PCA_FLAG)
    if PLOT_FLAG:
        plot_clusters(X,y,labels,score,CLUSTER_TYPE,RUN_NAME)

    if PV_FLAG:
        permue,pv=calculate_pv(X,PARAM_TUNINING_FLAG[1],PCA_FLAG,score)
        print(f' PV for parameters {PARAM_TUNINING_FLAG[1]} is {pv} ')
        plot_clusters(permue,y,labels,score,CLUSTER_TYPE,f'{RUN_NAME}_{CLUSTER_TYPE}')


    if GROUPED_FLAG:
        grouped_labels_to_excel(X, X2,y, labels,RUN_NAME)






if __name__=='__main__':
    warnings.simplefilter("ignore")
    file=r"evoplasmidsori\data\only_above_tresh_40bp.xlsx"
    PCA_FLAG=False
    PV_FLAG=  False
    PLOT_FLAG=True
    CLUSTER_TYPE=''
    CLUSTER_TYPE_POOL=['GMM','HDB','SP','AF','DPCA','KM']
    PARAM_TUNINING_FLAG=[True,[4,'spherical','random_from_data',100,0.0325,0.775,False]] # [True,None] | [False ,[eps=4,min_samples=7]]
    GROUPED_FLAG=True
    RUN_NAME='permu_thresh_40bp_rnap'
    run(file,PCA_FLAG,PV_FLAG,PARAM_TUNINING_FLAG,PLOT_FLAG,CLUSTER_TYPE,GROUPED_FLAG,RUN_NAME,CLUSTER_TYPE_POOL)
