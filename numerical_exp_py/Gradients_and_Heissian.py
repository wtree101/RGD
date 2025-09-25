import numpy as np
import math
default_regularization = 0.01

def get_gradient_BM_loss(X, measurements, ground_truth, regularization=default_regularization, stochastic=False, batch_size=10):
    num_measurements = measurements.shape[0]
    # Exhaustive computation for full batch size update
    if not stochastic:
        current_measurements = np.sum(measurements * (X @ X.T), axis=(1, 2))
        ground_truth_measurements = np.sum(measurements * ground_truth, axis=(1, 2)) 
        diff = current_measurements - ground_truth_measurements    
        #diff_alt=0
        #for i in range(num_measurements):
            #diff_alt += measurements[i] * diff[i]
        #gradient = diff_alt @ X
        gradient = np.einsum('k,kij->ij', diff, measurements) @ X
    
    # If stochastic, compute gradient on a mini-batch
    else:
        batch_indices = np.random.choice(num_measurements, size=min(batch_size, num_measurements), replace=False)
        batched_measurements = measurements[batch_indices]
        ground_truth_measurements = np.sum(batched_measurements * ground_truth, axis=(1, 2)) 
        current_measurements = np.sum(batched_measurements * (X @ X.T), axis=(1, 2))
        diff = current_measurements - ground_truth_measurements
        gradient = np.einsum('k,kij->ij', diff, batched_measurements) @ X
        num_measurements = len(batch_indices)
    return 4 * gradient / num_measurements + 2 * regularization * X

def get_gradient_convex_loss(M, measurements, ground_truth, regularization=default_regularization):
    num_measurements = measurements.shape[0]
    ground_truth_measurements = np.sum(measurements * ground_truth, axis=(1, 2)) 
    current_measurements = np.sum(measurements * M, axis=(1, 2))
    diff = current_measurements - ground_truth_measurements
    
    # Vectorized computation: sum over measurements weighted by diff
    gradient = np.einsum('k,kij->ij', diff, measurements) 
    return 2 * gradient / num_measurements


def get_gradient_BM_loss_l1mimic(X, measurements, ground_truth, regularization=default_regularization, stochastic=False, batch_size=10):
    num_measurements = measurements.shape[0]
    # Exhaustive computation for full batch size update
    if not stochastic:
        current_measurements = np.sum(measurements * (X @ X.T), axis=(1, 2))
        ground_truth_measurements = np.sum(measurements * ground_truth, axis=(1, 2)) 
        diff = current_measurements - ground_truth_measurements    
        #diff_alt=0
        #for i in range(num_measurements):
            #diff_alt += measurements[i] * diff[i]
        #gradient = diff_alt @ X
        gradient = np.einsum('k,kij->ij', diff, measurements) @ X
        #Now we rescale each entry of the gradient by the absolute value of the corresponding difference respactively
        gradient = gradient * np.abs(diff) / (np.abs(diff) + 1e-8)

    
    # If stochastic, compute gradient on a mini-batch
    else:
        batch_indices = np.random.choice(num_measurements, size=min(batch_size, num_measurements), replace=False)
        batched_measurements = measurements[batch_indices]
        ground_truth_measurements = np.sum(batched_measurements * ground_truth, axis=(1, 2)) 
        current_measurements = np.sum(batched_measurements * (X @ X.T), axis=(1, 2))
        diff = current_measurements - ground_truth_measurements
        gradient = np.einsum('k,kij->ij', diff, batched_measurements) @ X
        num_measurements = len(batch_indices)
        gradient = gradient * np.abs(diff) / (np.abs(diff) + 1e-8)
    return 4 * gradient / num_measurements + 2 * regularization * X
