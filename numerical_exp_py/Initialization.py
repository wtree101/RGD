import numpy as np

default_dimension = 20
default_rank = 5
default_search_rank = 20

def initialization_measurements_perfect(num_measurements=1, size=default_dimension, rescale=0):
    num_measurements = size * size
    measurements = np.zeros((num_measurements, size, size))
    for k in range(num_measurements):
        i = k // size
        j = k % size
        measurements[k, i, j] = 1
    return measurements

def initialization_measurements_prGaus(num_measurements=1, size=default_dimension, rescale=False):
    measurements = np.zeros((num_measurements, size, size))
    for k in range(num_measurements):
        measure_vector = np.random.normal(0, 1, size=(1, size))
        measurements[k] = measure_vector.T @ measure_vector
    if rescale: 
        measurements = measurements / np.linalg.norm(measurements, ord='fro', axis=(1, 2), keepdims=True)
        measurements = measurements * rescale
    return measurements

def initialization_measurements_Ginibre(num_measurements=1, size=default_dimension, rescale=0):
    measurements = np.zeros((num_measurements, size, size))
    for k in range(num_measurements):
        measurements[k] = np.random.randn(size, size) 
    if rescale: 
        measurements = measurements / np.linalg.norm(measurements, ord='fro', axis=(1, 2), keepdims=True)
        measurements = measurements * rescale
    return measurements

def ground_truth(size=default_dimension, rank=default_rank, symmetric=True):
    U = np.random.randn(size, rank)
    V = np.random.randn(size, rank)
    if symmetric:
        return U @ U.T 
    else:
        return U @ V.T 

def initialization_X(size=default_dimension, search_rank=default_search_rank, rescale=False):
    X = np.random.randn(size, search_rank)
    if rescale:
        X = X / np.linalg.norm(X, ord='fro')
        X = X * rescale
    return X