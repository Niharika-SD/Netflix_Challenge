from sklearn import metrics
from sklearn.cluster import KMeans,SpectralClustering
import scipy.io as sio
import os,sys
import numpy as np
from matplotlib import pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.decomposition import PCA

os.chdir('/home/niharika-shimona/Documents/')
colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

data = sio.loadmat('/home/niharika-shimona/Documents/Projects/results/A1L.mat')
X = data['A']
y = data['y_act']

n_labels = len(np.unique(y))
labels = np.ravel(y)
print labels.shape


t0 = time.time()

estimator = SpectralClustering(n_clusters=n_labels,affinity="rbf", n_init=10,n_neighbors = 6, assign_labels='kmeans')

estimator.fit(X.T)

t1 = time.time()

print("Spectral Clustering run \n")
print(79 * '_')
print('% 9s' % 'init' '    time   homo   compl  v-meas     ARI AMI  silhouette')
print('% 9s  %.4fs  %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % ('E frob dis', (t1 - t0),
             metrics.homogeneity_score(labels, estimator.labels_), 
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(X.T, estimator.labels_,
                                      metric='euclidean')))
y_pred = estimator.labels_.astype(np.int)
print y_pred

sio.savemat('/home/niharika-shimona/Documents/Projects/Netflix_Challenge/y_pred.mat', {'y_pred': y_pred})