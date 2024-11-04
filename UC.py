import numpy as np

def update_centroids(data, num_clusters, assignments):
    centroids = []                                                              #10 klastrów (jedna dla każdej cyfry od 0 do 9) i 64 cechy (piksele w obrazie 8x8), centroids będzie mieć wymiary 10 × 64
    for c in range(num_clusters):                                               #Zmienna c to indeks aktualnie iterowanego klastra, jeśli num_clusters wynosi 10, to c przyjmie wartości od 0 do 9.
        cluster_points = data[np.array(assignments) == c]                       #[Filtracja punktów danych do klastra] np.array(assignments) konwertuje listę assignments na tablicę NumPy; == c tworzy tablicę boolean (TvF), w której każda pozycja wskazuje, czy dany punkt został przypisany do klastra o indeksie c; otrzymujemy nową tablicę cluster_points, która zawiera tylko te punkty z data, które zostały przypisane do klastra o indeksie c.
        if len(cluster_points) > 0:                                             #Sprawdzenie, czy są punkty w klastrze
            centroids.append(np.mean(cluster_points, axis=0))                   #Obliczenie nowego centroidu. oblicza nowy centroid jako średnią dla tych punktów - Oblicza średnią dla wszystkich punktów w cluster_points wzdłuż osi 0 i Dodaje nowo obliczoną średnią (centroid) do listy centroids
        else:                                                                   #Jeśli nie ma punktów w klastrze to generujemy losowy centroid
            centroids.append(np.random.random(data.shape[1]))
    return np.array(centroids)