import numpy as np
from Initialization import initialization_measurements_Ginibre, ground_truth, initialization_X
from Optimization import optimization_BM_loss
import math

# In this document we will run the optimization process with preset parameters and determine the working sample size. 
# For simplicity, we will always assume that the ground truth is symmetric with rank 5, and the sample complexity grows from 500.
# In the case with the smallest search rank, the critical interval for sample size is [475, 500].



def convergence_check(search_rank, regularization, tolerance, step, trials, num_measurements=500, size=50, rank=5, default_rescale=1, nesterov=False, momentum=1, max_count=10000, stochastic=False, batch_size=10, grace=0):
    measurements = initialization_measurements_Ginibre(num_measurements, size, rescale=default_rescale)
    ground_truth_mat = ground_truth(size, rank, symmetric=True)

    # Run optimization
    results = []
    success = 0
    for _ in range(trials):
        X_init = initialization_X(size, search_rank, rescale=False)
        error, optimized_X, count = optimization_BM_loss(            
            X_init, 
            measurements, 
            ground_truth_mat,
            step_size=step,
            error_tolerance=tolerance,
            regularization=regularization,
            max_count=max_count,
            nesterov=nesterov,
            momentum=momentum,
            stochastic=stochastic,
            batch_size=batch_size,
            grace=grace

        )
        results.append((error, optimized_X, count))
        if error <= tolerance: 
            success += 1
        #check if there are more than 50% of successful trials
        if success > trials / 2:
            success = True
            break
        elif _ - success + 1 >= float(trials) / 2:
            success = False
            break 
    if success:
        error = np.mean([error for error, _, _ in results])
        count = np.mean([count for _, _, count in results])
        optimized_X = np.mean([optimized_X for _, optimized_X, _ in results], axis=0)
    else:
        error = np.mean([error for error, _, _ in results])
        count = None
        optimized_X = X_init

    return success, error, optimized_X, count

    
### The following function will attempt to find the optimal sample size for a given search rank and regularization parameter.
def sample_size_experiment(search_rank, regularization, tolerance, step, trials, lb, ub, jump=100, size=50, rank=5, default_rescale=1, nesterov=False, momentum=1, max_count=10000, stochastic=False, batch_size=10, grace=0):
    num_measurements_list = [i for i in range(lb, ub+jump, jump)]
    
    for num_measurements in num_measurements_list:
        success, error, optimized_X, count = convergence_check(search_rank, regularization, tolerance, step, trials, num_measurements=num_measurements, size=size, rank=rank, default_rescale=default_rescale, nesterov=nesterov, momentum=momentum, max_count=max_count, stochastic=stochastic, batch_size=batch_size, grace=grace)
        actual_search_rank = optimized_X.shape[1]
        if success:
            print(f"Search rank {actual_search_rank} at sample size {num_measurements} succeeded with error {error}, iterations {count}")
            return num_measurements, error, count
        print(f"Search rank {actual_search_rank} failed at sample size {num_measurements} with error {error}, iterations {count}")
    print(f"Search rank {actual_search_rank} No successful sample size found in the given range.")
    return 0, error, count

# Fixed parameters: reasonable tolerance and step size
# sample_size_experiment(5, 0, 0.6, 0.5, 7, 500, 700, jump=100, size=50, rank=5, default_rescale=1, nesterov=False, momentum=1, max_count=100)