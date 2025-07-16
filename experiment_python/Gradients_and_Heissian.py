import numpy as np

default_regularization = 0.01

def get_gradient_BM_loss(X, measurements, ground_truth, regularization=default_regularization):
    num_measurements = measurements.shape[0]
    ground_truth_measurements = np.sum(measurements * ground_truth, axis=(1, 2)) 
    current_measurements = np.sum(measurements * (X @ X.T), axis=(1, 2))
    diff = current_measurements - ground_truth_measurements
    gradient = 0
    
    for k in range(num_measurements):
        gradient += diff[k] * measurements[k] 
    gradient = gradient @ X
    return 4 * gradient / num_measurements + 2 * regularization * X

def get_gradient_convex_loss(M, measurements, ground_truth, regularization=default_regularization):
    num_measurements = measurements.shape[0]
    ground_truth_measurements = np.sum(measurements * ground_truth, axis=(1, 2)) 
    current_measurements = np.sum(measurements * M, axis=(1, 2))
    diff = current_measurements - ground_truth_measurements
    gradient = 0
    
    for k in range(num_measurements):
        gradient += diff[k] * measurements[k] 
    return 2 * gradient / num_measurements


