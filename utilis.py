import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler as SS
import umap
from matplotlib import pyplot as plt
from kneed import KneeLocator
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture
import time
from sklearn.preprocessing import normalize
from scipy import sparse
from collections import Counter


# colormap for ploting
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
    
# Embedding of each chunk
def Dimensionality_reduction(chunks, n_components):
    DR_chunks_g = []

    for chunk in chunks:
        reducer = umap.UMAP(n_components=n_components, metric='cosine',
                            n_neighbors=15, min_dist=0.1).fit_transform(
                            chunk)  # metric correlation is also working better
        scaler = MinMaxScaler().fit_transform(reducer)
        minma = normalize(scaler, norm='l2')

        plt.imshow(scaler.reshape(55, 56, 3))
        plt.axis('off')
        plt.show()
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(scaler[:, 0], scaler[:, 1], scaler[:, 2], c=minma, s=0.5)
        ax.grid(False)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])
        # ax.set_title('UMAP Embedding of Chunk')
        plt.show()
        DR_chunks_g.append(scaler)

    return DR_chunks_g

def f_ensemble_clustering(chunks_g):
    ensemble_clusterings = []
    est_optimal_clusters_list = []

    for i, chunk in enumerate(chunks_g):

        n_components = np.arange(2, 7) #Choose an appropriate range according to the data
        models = [GaussianMixture(n, max_iter=50, random_state=None, n_init=10,
                                  covariance_type='full').fit(chunk) for n in n_components]
        bic_scores = [m.bic(chunk) for m in models]

        #Plot BIC scores for the current chunk
        plt.plot(n_components, bic_scores)
        plt.title(f'BIC score of chunk {i + 1}')
        plt.xlabel('Number of clusters')
        plt.ylabel('BIC score')
        plt.close()
        print(n_components, 'k cluster')

        kneedle_point = KneeLocator(n_components, bic_scores, curve='convex', direction='decreasing')
        elbow_idx = np.where(n_components == kneedle_point.knee)[0][0]
        plt.plot(n_components[elbow_idx], kneedle_point.knee_y, 'o', markersize=7, label=f'Estimated K-Clusters')
        plt.legend(loc='best')
        plt.title(f'The suggested number of clusters = ' + np.str(kneedle_point.knee))
        plt.show()
        plt.close()
        print(elbow_idx, 'est-clus')
        est_optimal_n_clusters = n_components[elbow_idx]
        est_optimal_clusters_list.append(est_optimal_n_clusters)
        print(optimal_n_clusters, "bic clus")
        # get the clusters centers via GMM
        gmm = GaussianMixture(n_components=optimal_n_clusters, max_iter=10, covariance_type='full',
                              random_state=None, n_init=20)
        gmm.fit(chunk)
        gmm_cluster_centers = gmm.means_
        # Use KMeans with GMM cluster centers as initial starting point for clustering
        kmeans = KMeans(n_clusters=optimal_n_clusters, init=gmm_cluster_centers)
        GK_cluster_labels = kmeans.fit_predict(chunk)
        # plt.imshow(final_cluster_labels.reshape(image_shape), cmap)
        # plt.title(f'kmeans of subset {i + 1}')
        # plt.show()
        # plt.close()
        # Append the cluster labels to the ensemble clusterings
        ensemble_clusterings.append(GK_cluster_labels)

    count_occurrences = Counter(optimal_clusters_list)
    most_common_n_clustrs = count_occurrences.most_common(1)[0][0]
    print(f"Most common number of clusters: {most_common_n_clustrs}")
    return ensemble_clusterings, most_common_n_clustrs

def coo_mat(ensemble_clusterings_m):
    co_mat = []
    # compute co-occurrence matrix to resovle the label matching issue
    pixel_label = ensemble_clusterings_m
    num_pixels = ensemble_clusterings_m[0].shape[0]
    co_matrix = np.zeros((num_pixels, num_pixels), dtype=np.uint8)

    threshold = 0.4  # threshold as maj voting
    for k, label in enumerate(ensemble_clusterings_m):
        StaTime = time.time()
        clusterid_list = np.unique(label)
        for clusterid in clusterid_list:
            itemidx = np.where(label == clusterid)[0]
            for i, x in enumerate(itemidx):
                ys = itemidx[i + 1:]
                co_matrix[x, ys] += 1
                co_matrix[ys, x] += 1
        SpenTime = time.time() - StaTime
        print('running time for {}th label: {}'.format(k, round(SpenTime, 2)))

    threshold_count = threshold * np.max(co_matrix)

    # Set entries below the threshold count to 0
    co_matrix[co_matrix < threshold_count] = 0

    co_matrix = sparse.coo_matrix(co_matrix)
    print(co_matrix.shape, 'co-shape')

    co_mat = co_matrix
    return co_mat
