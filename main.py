#import libs
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, Birch
import matplotlib.cm as cm
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
import despike
from scipy.stats.mstats import spearmanr
from utilis import Dimensionality_reduction
from utilis import f_ensemble_clustering
from utilis import coo_mat
from utilis import find_nearest
from utilis import discrete_cmap


#color map for visualization
cmap = discrete_cmap(18, 'jet')

#data shape 
image_shape = 55, 56  # 51, 56 #HCRAC
#image_shape = 285, 260 #brain
#image_shape = 202, 107 #mouse fetus

# Data load
data = np.loadtxt("HCA_new_data.txt")
#data = np.loadtxt("brain_new_data.txt") #brain-data
#data = np.loadtxt('data_fe_2_update')
print(data.shape)

# generate n-chunks for original data
n_chunks = 5  # can be varied according to the complexity of data structure
chunk_size = int(data.shape[1] * 0.80) 
chunks = []
for _ in range(n_chunks):
    subset_indices = np.random.choice(data.shape[1], size=chunk_size, replace=False)
    chunk = data[:, subset_indices]
    chunks.append(chunk)
for m in chunks:
    print(np.array(m).shape)

#Reduce the dimensions of each chunk
DR_chunks = Dimensionality_reduction(chunks, n_components=3)
#get ensemble clustering
e_clusterings, k_clustersss = f_ensemble_clustering(DR_chunks)
print(k_clustersss, 'k-clusters')
print("Ensemble Clustering:", e_clusterings)
print("Length of ensemble_clusterings:", len(e_clusterings))
labels_arrayA = np.array(e_clusterings)
print(labels_arrayA.shape, 'out-shape')

###########################
#compute the co-occurance matric of ensemble clustering#
# and reduce it for faster computation #
###########################
co_matrix = coo_mat(e_clusterings)
# To reduce in order to faster the computation of hierarchy
#M_PCs = 50
SVD_threshold = 0.99
svd = TruncatedSVD(n_components=51, n_iter=50)
pcs_all = svd.fit_transform(co_matrix)
evr = svd.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)
cut_evr = find_nearest(evr_cumsum, SVD_threshold)
nPCs = np.where(evr_cumsum == cut_evr)[0][0] + 1
if nPCs >= 50:
    nPCs = 50
new_pc = pcs_all[:, :nPCs]
print(new_pc.shape, 'new-pc')

# Get the final clustering assignments of majority co-occurrence
p_clus = sch.linkage(pcs_all[:, :nPCs], method='complete', metric='cosine')
print(p_clus.shape, 'Y')

#n_clusters = 3
#t can either be distance or clusters
predict_clustering = sch.fcluster(p_clus, t=k_clustersss, criterion='maxclust')-1 #t is auto computed
print(predict_clustering.shape, 'ind')
plt.imshow(predict_clustering.reshape(image_shape), cmap)
plt.title('Seg-Map')
#plt.savefig('voting-eLIMS')
plt.show()
plt.close()

###########################
#cleaning the mild noise if any#
###########################
img = despike.spikes(predict_clustering.reshape(image_shape))
#X_pred1 = img.reshape(13)
plt.title('n-Spikes', fontsize=24)
plt.imshow(img, cmap=cmap)
plt.show()
plt.close()

img1 = despike.clean(predict_clustering.reshape(image_shape))
#X_pred = img1.reshape(202, 107)
plt.imshow(img1, cmap=cmap)
plt.axis('off')
plt.show()

###########################
#Correlate m/z ions for the choosen ROI#
###########################
#load the related data #
data_org = np.loadtxt("HCA_new_data.txt") # load the original data

All_mz = np.loadtxt("peak_list_HCA_a") # original m/zs
print(All_mz.shape, "total number of mz")

#labels = np.load('elims-HCA-majVoting.npy')
labels = predict_clustering
print(labels.shape, 'predicted labels shape')

#choose the label for target ROI 
for i in range(len(list(set(labels)))):
    show = np.where(labels == i, 1, 0)
    show = show.reshape(image_shape)
    plt.title("Cluster number " + str(i))

    plt.imshow(show)
    plt.axis('off')
    plt.show()

# Correlate the Select ROI with the mzPeaks:
cluster_id = 0
ROI = labels == cluster_id
ROI = ROI.astype(int)

MSI_Pt = data_org #get MSI data only for all org. m/z peaks
Corr_Val = np.zeros(len(All_mz))
for i in range(len(All_mz)):
    Corr_Val[i] = spearmanr(ROI, MSI_Pt[:, i])[0]
id_mzCorr = np.argmax(Corr_Val)
rank_ij = np.argsort(Corr_Val)[::-1]
x = All_mz[id_mzCorr]
im = MSI_Pt[:, id_mzCorr]
im = im.reshape(image_shape)
plt.imshow(im)
plt.title("m/z " + str(x))
plt.axis('off')
plt.show()
print('m/z', All_mz[id_mzCorr])
print('corr_Value = ', Corr_Val[id_mzCorr])
#plt.plot(All_mz,Corr_Val)
plt.show()
print(['%0.4f' % i for i in All_mz[rank_ij[0:10]]])
print('Correlation Top 10 Ranked peaks:', end='')
print(['%0.4f' % i for i in Corr_Val[rank_ij[0:10]]])

