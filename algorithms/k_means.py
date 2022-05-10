from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


def algorithm_k_means(temp):
    """
    将传入的数据进行K_MEANS++聚类，并可视化
    :param temp: type: np.array
    :return: labels
    """
    kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
    labels = kmeans.fit(np.matrix(temp.values))
    plt.scatter(temp.values[:, 0], temp.values[:, 1], c=labels.labels_, cmap='rainbow')
    plt.show()
