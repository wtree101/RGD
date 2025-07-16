import numpy as np
from Gradients_and_Heissian import get_gradient_BM_loss, get_gradient_convex_loss

def update_BM_gradient(X, measurements, ground_truth, step_size, gradient):
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


def optimization_convex_loss(X, measurements, ground_truth, step_size=0.01, 
                           regularization=0.01, error_tolerance=1e-6, max_count=500):
    error = 1
    count = 0 
    M = X @ X.T
    
    while error > error_tolerance:
        gradient = get_gradient_convex_loss(M, measurements, ground_truth, regularization)        
        M -= step_size * gradient    
        error = np.linalg.norm(M - ground_truth, 'fro')
        count += 1
        
        if count % 100 == 0 or count < 100:
            print(f"Iteration {count}, Error: {error}, Gradient: {np.linalg.norm(gradient)}")
        if count > max_count:
            print("Maximum iteration count reached without convergence")
            break
    return error, M

def optimization_BM_loss(X, measurements, ground_truth, step_size, 
                        regularization=0.01, error_tolerance=1e-2, max_count=500):
    error = error_tolerance + 1
    count = 0 
    
    while error > error_tolerance:
        gradient = get_gradient_BM_loss(X, measurements, ground_truth, regularization)    
        X = update_BM_gradient(X, measurements, ground_truth, step_size, gradient=gradient)
        error = np.linalg.norm(X @ X.T - ground_truth, 'fro')
        count += 1
        
        if count % 100 == 0 or count < 100:
            print(f"Iteration {count}, Error: {error}, Gradient: {np.linalg.norm(gradient)}")
        if count > max_count:
            print("Maximum iteration count reached without convergence")
            break
    return error, X, count