import numpy as np 
import matplotlib.pyplot as plt
import math

# Check NumPy version
print(f"NumPy version: {np.__version__}")

### utilities: initialization, gradient, plotting 
default_dimension=20 
default_rank=5
default_regularization=0.01
default_search_rank=20
def Initialization_measurements_perfect(num_measurements=1, size=default_dimension, rescale=0):
    num_measurements = size * size
    measurements = np.zeros((num_measurements, size, size))
    for k in range(num_measurements):
        i = k // size
        j = k % size
        measurements[k, i, j] = 1
    return measurements



def Initialization_measurements_prGaus(num_measurements=1, size=default_dimension, rescale=False):
    measurements= np.zeros((num_measurements, size, size))
    for k in range(num_measurements):
        measurements[k] = np.random.randn(size) @ np.random.randn(size).T 
    if rescale: 
        measurements = measurements / np.linalg.norm(measurements, ord='fro', axis=(1, 2), keepdims=True)
        measurements = measurements*rescale
    return measurements

def Initialization_measurements_Ginibre(num_measurements=1, size=default_dimension, rescale=0):
    measurements= np.zeros((num_measurements, size, size))
    for k in range(num_measurements):
        measurements[k] = np.random.randn(size, size) 
    if rescale: 
        measurements = measurements / np.linalg.norm(measurements, ord='fro', axis=(1, 2), keepdims=True)
        measurements = measurements*rescale
    return measurements


def Ground_truth(size=default_dimension, rank=default_rank, symmetric=True):
    """Generate a random ground truth matrix."""
    U = np.random.randn(size, rank)
    V = np.random.randn(size, rank)
    if symmetric:
        return U @ U.T 
    else:
        return U @ V.T 

#Loss function: (1/n)\cdot \sum_{k=1}^{num_measurements} (<A_k, XX^T>^2  - <A_k, X_*X_*^T>^2)^2 + \norm{X}_F^2#
def Initialization_X(size=default_dimension, search_rank=default_search_rank, rescale=False):
    """Initialize the matrix X."""
    X = np.random.randn(size, search_rank)
    if rescale:
        X = X / np.linalg.norm(X, ord='fro')
        X = X * rescale
    return X


def Update_BM_gradient(X, measurements, ground_truth, step_size=0.01, gradient=0):
    # update the matrix X using a fixed step size
    if True: 
        X = X - step_size * gradient
    else: 
        Heissian=Get_Heissian_standard_loss(X, measurements, ground_truth)
        (S, M, M)= np.linalg.eig(Heissian)
        if np.all(S > 0):
            print("Warning: Second order critical point detected")
        else: 
            #Find the smallest eigenvalue and its corresponding eigenvector
            min_index = np.argmin(S)
            min_eigenvalue = S[min_index]
            min_eigenvector = M[:, min_index]
            # Update X using the eigenvector
            X = X - step_size * min_eigenvector.reshape(X.shape)
    return X

def Get_gradient_BM_loss(X, measurements, ground_truth, regularization=default_regularization):
    num_measurements = measurements.shape[0]
    ground_truth_measurements = np.sum(measurements * ground_truth, axis=(1, 2)) 
    current_measurements = np.sum(measurements * (X @ X.T), axis=(1, 2))
    # Fix the gradient calculation
    diff = current_measurements - ground_truth_measurements
    gradient = 0
    
    for k in range(num_measurements):
        gradient += diff[k] * measurements[k] 
    gradient = gradient @ X
    gradient = 4 * gradient / num_measurements + 2 * regularization * X
    return gradient
def Get_gradient_convex_loss(M, measurements, ground_truth, regularization=default_regularization):
    num_measurements = measurements.shape[0]
    ground_truth_measurements = np.sum(measurements * ground_truth, axis=(1, 2)) 
    current_measurements = np.sum(measurements * (M), axis=(1, 2))
    # Fix the gradient calculation
    diff = current_measurements - ground_truth_measurements
    gradient = 0
    
    for k in range(num_measurements):
        gradient += diff[k] * measurements[k] 
    gradient = 2 * gradient / num_measurements
    return gradient

def Get_Heissian_BM_loss(X, measurements, ground_truth, regularization=default_regularization):
    """Compute the Hessian of the standard loss function."""
    num_measurements = measurements.shape[0]



def Optimization_convex_loss(X, measurements, ground_truth, step_size=0.01, regularization=default_regularization, error_tolerance=1e-6, max_count=500):
    """Optimize the matrix X using the standard loss function."""
    num_measurements = measurements.shape[0]
    error=1
    count= 0 
    M = X @ X.T
    while error > error_tolerance:
        #gradient = Get_gradient_standard_loss(X, measurements, ground_truth, regularization)
        gradient = Get_gradient_convex_loss(M, measurements, ground_truth, regularization)        
        M=M-step_size * gradient    
        #X=Update_standard_gradient(X, measurements, ground_truth, step_size=0.001, gradient=gradient)    
        #error = np.linalg.norm(X @ (X.T) - ground_truth, ord='fro') 
        error=np.linalg.norm(M - ground_truth)
        count += 1
        if count % 100 == 0 or count < 100:
            print(f"Iteration {count}, Error: {error}, Gradient: {np.linalg.norm(gradient)}" )
        if count > max_count:
            print("Warning: Maximum iteration count reached without convergence.")
            break
 
    #print("final difference: ", X @ (X.T) - ground_truth)    
    print("Final M:", M)
    print("Final error:", error)
    return error, M 


def Optimization_BM_loss(X, measurements, ground_truth, step_size=0.01, regularization=default_regularization, error_tolerance=1e-2, max_count=500):
    """Optimize the matrix X using the standard loss function."""
    num_measurements = measurements.shape[0]
    error=1
    count= 0 
    while error > error_tolerance:
        gradient = Get_gradient_BM_loss(X, measurements, ground_truth, regularization)
        X=X-step_size * gradient    
        X=Update_BM_gradient(X, measurements, ground_truth, step_size=0.001, gradient=gradient)    
        error = np.linalg.norm(X @ (X.T) - ground_truth, ord='fro') 
        count += 1
        if count % 100 == 0 or count < 100:
            print(f"Iteration {count}, Error: {error}, Gradient: {np.linalg.norm(gradient)}" )
        if count > max_count:
            print("Warning: Maximum iteration count reached without convergence.")
            break
 
    return error, X, count

#Let's give it a try 
if __name__ == "__main__":
    num_measurements = 475
    size = 50
    rank = 5
    default_rescale =  1
    default_regularization = 0.001
    default_search_rank = 10
    measurements = Initialization_measurements_Ginibre(num_measurements, size, rescale=default_rescale)
    ground_truth = Ground_truth(size, rank, symmetric=True)
    X = Initialization_X(size, rank, rescale=1)

    error, optimized_X, count = Optimization_BM_loss(X, measurements, ground_truth, step_size=0.9, regularization=default_regularization, max_count=10000)
    print(f"Final error: {error}, Number of iterations: {count}")
    #print(f"Optimized X:\n{optimized_X}")