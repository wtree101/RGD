import numpy as np
import matplotlib.pyplot as plt
from Initialization import initialization_measurements_Ginibre, ground_truth, initialization_X
from Optimization import optimization_BM_loss, optimization_BM_loss_SGD, optimization_BM_loss_mini_batch
import time

def compare_optimization_methods():
    """
    Compare different optimization methods: Full batch GD, SGD, and Mini-batch GD
    """
    # Configuration
    num_measurements = 500
    size = 50
    rank = 5
    search_rank = 5
    step_size = 0.01
    regularization = 0.01
    error_tolerance = 1e-2
    max_count = 5000
    
    # Generate data
    measurements = initialization_measurements_Ginibre(num_measurements, size, rescale=1.0)
    ground_truth_mat = ground_truth(size, rank, symmetric=True)
    
    print(f"Problem setup: {size}x{size} matrix, rank {rank}, {num_measurements} measurements")
    print("=" * 70)
    
    # Test different methods
    methods = []
    
    # 1. Full batch gradient descent (original)
    print("1. Full Batch Gradient Descent:")
    X_init = initialization_X(size, search_rank, rescale=False)
    start_time = time.time()
    error_full, X_full, count_full = optimization_BM_loss(
        X_init.copy(), measurements, ground_truth_mat, 
        step_size=step_size, regularization=regularization, 
        error_tolerance=error_tolerance, max_count=max_count, 
        nesterov=False, report=False
    )
    time_full = time.time() - start_time
    methods.append(('Full Batch GD', error_full, count_full, time_full))
    print(f"   Error: {error_full:.6f}, Iterations: {count_full}, Time: {time_full:.2f}s")
    
    # 2. Stochastic Gradient Descent
    print("\n2. Stochastic Gradient Descent (batch_size=32):")
    start_time = time.time()
    error_sgd, X_sgd, count_sgd = optimization_BM_loss_SGD(
        X_init.copy(), measurements, ground_truth_mat, 
        step_size=step_size, regularization=regularization, 
        error_tolerance=error_tolerance, max_count=max_count, 
        batch_size=32, momentum=0.9, report=False
    )
    time_sgd = time.time() - start_time
    methods.append(('SGD (batch=32)', error_sgd, count_sgd, time_sgd))
    print(f"   Error: {error_sgd:.6f}, Iterations: {count_sgd}, Time: {time_sgd:.2f}s")
    
    # 3. Mini-batch Gradient Descent
    print("\n3. Mini-batch Gradient Descent (batch_size=64):")
    start_time = time.time()
    error_mb, X_mb, count_mb = optimization_BM_loss_mini_batch(
        X_init.copy(), measurements, ground_truth_mat, 
        step_size=step_size, regularization=regularization, 
        error_tolerance=error_tolerance, max_count=max_count, 
        batch_size=64, shuffle=True, report=False
    )
    time_mb = time.time() - start_time
    methods.append(('Mini-batch GD (batch=64)', error_mb, count_mb, time_mb))
    print(f"   Error: {error_mb:.6f}, Iterations: {count_mb}, Time: {time_mb:.2f}s")
    
    # 4. Different batch sizes for SGD
    print("\n4. SGD with different batch sizes:")
    batch_sizes = [8, 16, 32, 64, 128]
    sgd_results = []
    
    for batch_size in batch_sizes:
        start_time = time.time()
        error_sgd_bs, X_sgd_bs, count_sgd_bs = optimization_BM_loss_SGD(
            X_init.copy(), measurements, ground_truth_mat, 
            step_size=step_size, regularization=regularization, 
            error_tolerance=error_tolerance, max_count=max_count, 
            batch_size=batch_size, momentum=0.9, report=False
        )
        time_sgd_bs = time.time() - start_time
        sgd_results.append((batch_size, error_sgd_bs, count_sgd_bs, time_sgd_bs))
        print(f"   Batch size {batch_size:3d}: Error={error_sgd_bs:.6f}, Iterations={count_sgd_bs}, Time={time_sgd_bs:.2f}s")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    for method, error, count, time_elapsed in methods:
        print(f"{method:25s}: Error={error:.6f}, Iter={count:4d}, Time={time_elapsed:6.2f}s")
    
    return methods, sgd_results

def test_sgd_convergence():
    """
    Test SGD convergence behavior
    """
    print("\nTesting SGD convergence behavior...")
    
    # Configuration
    num_measurements = 300
    size = 30
    rank = 3
    search_rank = 3
    
    # Generate data
    measurements = initialization_measurements_Ginibre(num_measurements, size, rescale=1.0)
    ground_truth_mat = ground_truth(size, rank, symmetric=True)
    X_init = initialization_X(size, search_rank, rescale=False)
    
    # Test different step sizes
    step_sizes = [0.001, 0.01, 0.1, 0.5]
    
    print(f"Testing step sizes: {step_sizes}")
    for step_size in step_sizes:
        try:
            error_sgd, X_sgd, count_sgd = optimization_BM_loss_SGD(
                X_init.copy(), measurements, ground_truth_mat, 
                step_size=step_size, regularization=0.01, 
                error_tolerance=1e-2, max_count=2000, 
                batch_size=32, momentum=0.9, report=False
            )
            print(f"Step size {step_size:6.3f}: Error={error_sgd:.6f}, Iterations={count_sgd}")
        except Exception as e:
            print(f"Step size {step_size:6.3f}: Failed - {str(e)}")

if __name__ == "__main__":
    # Run comparison
    methods, sgd_results = compare_optimization_methods()
    
    # Test convergence
    test_sgd_convergence()
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)
    print("1. For large datasets: Use SGD with batch_size=32-64 for faster iterations")
    print("2. For stable convergence: Use mini-batch GD with batch_size=64-128")
    print("3. For best accuracy: Use full batch GD (your current method)")
    print("4. Consider adaptive learning rates for better convergence")
