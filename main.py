import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from UA import update_assignments
from UC import update_centroids


data, labels = load_digits(return_X_y=True)
(n_samples, n_features), n_digits = data.shape, np.unique(labels).size
print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")

num_clusters = n_digits

centroids = data[np.random.choice(len(data), num_clusters, replace=False)]      

iteration = 0
while True:
    iteration += 1
    previous_centroids = centroids.copy()                                       
    assignments = update_assignments(data, centroids)                           
    centroids = update_centroids(data, num_clusters, assignments)               

  
    if np.array_equal(centroids, previous_centroids):
        print(f"Centroidy przestały się zmieniać po {iteration} iteracjach.")
        break


pca = PCA(2)
data_2d = pca.fit_transform(data)
centroids_2d = pca.transform(centroids)

plt.scatter(data_2d[:, 0], data_2d[:, 1], c=assignments, cmap='viridis', s=5, label='Dane')
plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], color='red', marker='*', s=200, label='Centroidy')
plt.legend()
plt.show()

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
kmeans_ari = adjusted_rand_score(labels, assignments)
kmeans_nmi = normalized_mutual_info_score(labels, assignments)

print(f"k-means ARI: {kmeans_ari:.2f}, NMI: {kmeans_nmi:.2f}")

from sklearn.cluster import AgglomerativeClustering, DBSCAN

# Klasteryzacja hierarchiczna
agg = AgglomerativeClustering(n_clusters=num_clusters)
agg_assignments = agg.fit_predict(data)
agg_ari = adjusted_rand_score(labels, agg_assignments)
agg_nmi = normalized_mutual_info_score(labels, agg_assignments)
print(f"Agglomerative Clustering ARI: {agg_ari:.2f}, NMI: {agg_nmi:.2f}")

# Klasteryzacja DBSCAN
dbscan = DBSCAN(eps=15, min_samples=5)
dbscan_assignments = dbscan.fit_predict(data)

# DBSCAN zwraca -1 dla punktów uznanych za szum
# Konwertujemy to do postaci, która jest użyteczna do porównań
dbscan_assignments[dbscan_assignments == -1] = num_clusters  # Oznaczamy szum jako nowy klaster

dbscan_ari = adjusted_rand_score(labels, dbscan_assignments)
dbscan_nmi = normalized_mutual_info_score(labels, dbscan_assignments)
print(f"DBSCAN ARI: {dbscan_ari:.2f}, NMI: {dbscan_nmi:.2f}")


import unittest

class TestKMeansFunctions(unittest.TestCase):

    def test_update_assignments(self):
        data = np.array([[1, 2], [2, 3], [3, 4]])
        centroids = np.array([[1, 2], [4, 5]])
        assignments = update_assignments(data, centroids)
        expected_assignments = [0, 0, 1]  
        self.assertEqual(assignments, expected_assignments)

    def test_update_centroids(self):
        data = np.array([[1, 2], [2, 3], [3, 4]])
        assignments = [0, 0, 1]  
        new_centroids = update_centroids(data, 2, assignments)
        expected_centroids = np.array([[1.5, 2.5], [3, 4]])  #
        np.testing.assert_array_almost_equal(new_centroids, expected_centroids)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False) #albo pusty nawias, na collabie nie działa


