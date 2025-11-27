import numpy as np
from Gradients_and_Heissian import get_gradient_BM_loss, get_gradient_convex_loss, get_gradient_BM_loss_l1mimic
import math
from Initialization import initialization_measurements_Ginibre, ground_truth, initialization_X


def update_BM_gradient(X, measurements, ground_truth, step_size, gradient):
    # update the matrix X using a fixed step size
    if True: 
        X = X - step_size * gradient
    
    return X
    



def optimization_convex_loss(X, measurements, ground_truth, step_size=0.01, 
                           regularization=0.01, error_tolerance=1e-6, max_count=500, nesterov=True, report=True):
    error = 1
    count = 0 
    M = X @ X.T
    
    while error > error_tolerance:
        gradient = get_gradient_convex_loss(M, measurements, ground_truth, regularization)        
        M -= step_size * gradient    
        error = np.linalg.norm(M - ground_truth, 'fro')
        count += 1
        
        if count % 100 == 0 and report:
            print(f"Iteration {count}, Error: {error}, Gradient: {np.linalg.norm(gradient)}")
        if count > max_count:
            print("Maximum iteration count reached without convergence")
            break
    return error, M

def optimization_BM_loss(X, measurements, ground_truth, step_size, 
                        regularization=0.01, error_tolerance=1e-2, max_count=500, nesterov=False, momentum=1, report=True, grace=2, stochastic=False, batch_size=10):
    error_document=np.zeros(max_count)
    error = error_tolerance + 1
    count = 0 
    X_prev = X.copy()
    lambda_=lambda_prev=1
    grace = grace
    while error > error_tolerance:
        if nesterov:
            # Nesterov's acceleration
            lambda_=(1+math.sqrt(1+4*(lambda_prev**2)))/2         
            omega_= (lambda_prev - 1)/(lambda_)
            lambda_prev = lambda_   
            Y = X + momentum*omega_ * (X - X_prev) 
            X_prev = X.copy()
            gradient = get_gradient_BM_loss(Y, measurements, ground_truth, regularization, stochastic=stochastic, batch_size=batch_size)        
            X = X - step_size * gradient
        else: 
            gradient = get_gradient_BM_loss(X, measurements, ground_truth, regularization, stochastic=stochastic, batch_size=batch_size)        
            X = X - step_size * gradient
            
        error = np.linalg.norm(X @ X.T - ground_truth, 'fro')
        error_document[count] = error
        count += 1
        
        if count % 100 == 0 and report:
            print(f"Iteration {count}, Error: {error}, Gradient: {np.linalg.norm(gradient)}")
        if count >= max_count:
            if grace and error < error_tolerance + 0.4:
                print("Maximum iteration count reached, but error is close to tolerance, continuing with grace period")
                max_count += 5000
                grace-=1
                continue    
            print("Maximum iteration count reached without convergence")
            break
            

    return error, X, count, error_document

def optimization_BM_loss_altreg(X, measurements, ground_truth, step_size, 
                        regularization=0.01, error_tolerance=1e-2, max_count=500, nesterov=False, momentum=1, report=True, grace=2, stochastic=False, batch_size=10, threshold_1=0, threshold_2=0, threshold_3=0, alt_regularization_1=0, alt_regularization_2=0, alt_regularization_3=0):
    error_document=np.zeros(max_count+1)
    error = error_tolerance + 1000
    count = 0 
    X_prev = X.copy()
    lambda_=lambda_prev=1
    grace = grace
    step_adj=True
    while error > error_tolerance:
        if error < threshold_1:
            regularization =alt_regularization_1
        if error < threshold_2:
            regularization = alt_regularization_2
        if error < threshold_3:
            if step_adj:
                regularization = alt_regularization_3
                step_adj = False
            elif regularization > 0.0001:
                regularization -= alt_regularization_3*0.0001
        if nesterov:
            # Nesterov's acceleration
            lambda_=(1+math.sqrt(1+4*(lambda_prev**2)))/2         
            omega_= (lambda_prev - 1)/(lambda_)
            lambda_prev = lambda_   
            Y = X + momentum*omega_ * (X - X_prev) 
            X_prev = X.copy()
            gradient = get_gradient_BM_loss(Y, measurements, ground_truth, regularization, stochastic=stochastic, batch_size=batch_size)        
            X = X - step_size * gradient
        else: 
            gradient = get_gradient_BM_loss(X, measurements, ground_truth, regularization, stochastic=stochastic, batch_size=batch_size)        
            X = X - step_size * gradient
        error = np.linalg.norm(X @ X.T - ground_truth, 'fro')/X.shape[0]
        error_document[count] = error
        count += 1
        
        if count % 100 == 0 and report:
            print(f"Iteration {count}, Error: {error}, Gradient: {np.linalg.norm(gradient)}")
        if count > max_count:
            if grace and error < error_tolerance + 0.4:
                print("Maximum iteration count reached, but error is close to tolerance, continuing with grace period")
                max_count += 5000
                grace-=1
                continue    
            print("Maximum iteration count reached without convergence")
            break

    return error, X, count, error_document
