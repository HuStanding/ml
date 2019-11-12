from time import time
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.cluster import adjusted_mutual_info_score
import matplotlib.pyplot as plt
# 生成样本点
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels = make_blobs(n_samples=750, centers=centers,
                        cluster_std=0.4, random_state=0)
# 可视化聚类结果
def plot_clustering(X, labels, title=None):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='prism')
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout()
    
# 进行 Agglomerative 层次聚类
linkage_method_list = ['single', 'complete', 'average', 'ward']
plt.figure(figsize=(10, 8))
ncols, nrows = 2, int(np.ceil(len(linkage_method_list) / 2))
plt.subplots(nrows=nrows, ncols=ncols)
for i, linkage_method in enumerate(linkage_method_list):
    print('method %s:' % linkage_method)
    start_time = time()
    Z = linkage(X, method=linkage_method)
    labels_pred = fcluster(Z, t=10, criterion='maxclust')
    print('Adjust mutual information: %.3f' %
            adjusted_mutual_info_score(labels, labels_pred))
    print('time used: %.3f seconds' % (time() - start_time))
    plt.subplot(nrows, ncols, i + 1)
    plot_clustering(X, labels_pred, '%s linkage' % linkage_method)
plt.show()


def hierarchical(dataSet, method = "complete", metric='euclidean', eps = 0.9):
    """
    @brief      层次聚类
    @param      dataSet  The data set
    @param      method   The method
    @param      metric   The metric
    @return     { description_of_the_return_value }
    """
    Z = linkage(dataSet, method=method, metric = metric)
    labels_pred = fcluster(Z, t=eps, criterion='maxclust')
    return labels_pred
    