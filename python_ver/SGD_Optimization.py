import numpy as np
from Gradients_and_Heissian import get_gradient_BM_loss, get_gradient_convex_loss
import math
import random

class SGDOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9, batch_size=32):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.velocity = None
        
    def get_stochastic_gradient(self, X, measurements, ground_truth, regularization, batch_indices=None):
        """
        Compute stochastic gradient using a mini-batch of measurements
        """
        num_measurements = measurements.shape[0]
        
        if batch_indices is None:
            # Randomly sample batch indices
            batch_indices = np.random.choice(num_measurements, 
                                           size=min(self.batch_size, num_measurements), 
                                           replace=False)
        
        # Get mini-batch
        batch_measurements = measurements[batch_indices]
        
        # Compute gradient on mini-batch
        ground_truth_measurements = np.sum(batch_measurements * ground_truth, axis=(1, 2))
        current_measurements = np.sum(batch_measurements * (X @ X.T), axis=(1, 2))
        diff = current_measurements - ground_truth_measurements
        
        # Vectorized computation for mini-batch
        gradient = np.einsum('k,kij->ij', diff, batch_measurements) @ X
        
        # Scale by batch size and add regularization
        gradient = 4 * gradient / len(batch_indices) + 2 * regularization * X
        
        return gradient
    
    def update(self, X, gradient):
        """
        Update parameters using SGD with momentum
        """
        if self.velocity is None:
            self.velocity = np.zeros_like(X)
        
        # Momentum update
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradient
        X_new = X + self.velocity
        
        return X_new

def optimization_BM_loss_SGD(X, measurements, ground_truth, step_size=0.01, 
                            regularization=0.01, error_tolerance=1e-2, max_count=500, 
                            batch_size=32, momentum=0.9, report=True):
    """
    Optimize using Stochastic Gradient Descent
    """
    error = error_tolerance + 1
    count = 0
    
    # Initialize SGD optimizer
    optimizer = SGDOptimizer(learning_rate=step_size, momentum=momentum, batch_size=batch_size)
    
    while error > error_tolerance:
        # Get stochastic gradient
        gradient = optimizer.get_stochastic_gradient(X, measurements, ground_truth, regularization)
        
        # Update using SGD
        X = optimizer.update(X, gradient)
        
        # Compute error
        error = np.linalg.norm(X @ X.T - ground_truth, 'fro')
        count += 1
        
        if count % 100 == 0 and report:
            print(f"Iteration {count}, Error: {error}, Gradient norm: {np.linalg.norm(gradient)}")
        
        if count > max_count:
            print("Maximum iteration count reached without convergence")
            break
    
    return error, X, count

def optimization_BM_loss_mini_batch(X, measurements, ground_truth, step_size=0.01, 
                                   regularization=0.01, error_tolerance=1e-2, max_count=500, 
                                   batch_size=32, shuffle=True, report=True):
    """
    Optimize using mini-batch gradient descent with shuffling
    """
    error = error_tolerance + 1
    count = 0
    num_measurements = measurements.shape[0]
    
    while error > error_tolerance:
        # Create batch indices
        indices = np.arange(num_measurements)
        if shuffle:
            np.random.shuffle(indices)
        
        # Process mini-batches
        for i in range(0, num_measurements, batch_size):
            batch_indices = indices[i:i+batch_size]
            
            # Get mini-batch gradient
            batch_measurements = measurements[batch_indices]
            ground_truth_measurements = np.sum(batch_measurements * ground_truth, axis=(1, 2))
            current_measurements = np.sum(batch_measurements * (X @ X.T), axis=(1, 2))
            diff = current_measurements - ground_truth_measurements
            
            # Compute gradient
            gradient = np.einsum('k,kij->ij', diff, batch_measurements) @ X
            gradient = 4 * gradient / len(batch_indices) + 2 * regularization * X
            
            # Update
            X = X - step_size * gradient
            
            count += 1
            
            if count % 100 == 0 and report:
                error = np.linalg.norm(X @ X.T - ground_truth, 'fro')
                print(f"Iteration {count}, Error: {error}, Gradient norm: {np.linalg.norm(gradient)}")
            
            if count > max_count:
                print("Maximum iteration count reached without convergence")
                return error, X, count
        
        # Compute error after each epoch
        error = np.linalg.norm(X @ X.T - ground_truth, 'fro')
    
    return error, X, count

def optimization_BM_loss_adaptive_SGD(X, measurements, ground_truth, step_size=0.01, 
                                     regularization=0.01, error_tolerance=1e-2, max_count=500, 
                                     batch_size=32, beta1=0.9, beta2=0.999, eps=1e-8, report=True):
    """
    Optimize using Adam-like adaptive SGD
    """
    error = error_tolerance + 1
    count = 0
    
    # Initialize Adam parameters
    m = np.zeros_like(X)  # First moment
    v = np.zeros_like(X)  # Second moment
    
    while error > error_tolerance:
        # Get stochastic gradient
        num_measurements = measurements.shape[0]
        batch_indices = np.random.choice(num_measurements, 
                                       size=min(batch_size, num_measurements), 
                                       replace=False)
        
        batch_measurements = measurements[batch_indices]
        ground_truth_measurements = np.sum(batch_measurements * ground_truth, axis=(1, 2))
        current_measurements = np.sum(batch_measurements * (X @ X.T), axis=(1, 2))
        diff = current_measurements - ground_truth_measurements
        
        gradient = np.einsum('k,kij->ij', diff, batch_measurements) @ X
        gradient = 4 * gradient / len(batch_indices) + 2 * regularization * X
        
        # Adam update
        count += 1
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient ** 2)
        
        # Bias correction
        m_hat = m / (1 - beta1 ** count)
        v_hat = v / (1 - beta2 ** count)
        
        # Update
        X = X - step_size * m_hat / (np.sqrt(v_hat) + eps)
        
        error = np.linalg.norm(X @ X.T - ground_truth, 'fro')
        
        if count % 100 == 0 and report:
            print(f"Iteration {count}, Error: {error}, Gradient norm: {np.linalg.norm(gradient)}")
        
        if count > max_count:
            print("Maximum iteration count reached without convergence")
            break
    
    return error, X, count
