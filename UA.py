import numpy as np

def update_assignments(data, centroids):
    assignments = []                                                            
    for point in data:                                                          
        distances = np.linalg.norm(point - centroids, axis=1)                   
        assignments.append(np.argmin(distances))                                
    return assignments

