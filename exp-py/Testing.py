import numpy as np
from Initialization import initialization_measurements_Ginibre, ground_truth, initialization_X, initialization_measurements_prGaus
from Optimization import optimization_BM_loss, optimization_BM_loss_altreg
import math

# In this document we will run the optimization process with preset parameters and determine the working sample size. 
# For simplicity, we will always assume that the ground truth is symmetric with rank 5, and the sample complexity grows from 500.
# In the case with the smallest search rank, the critical interval for sample size is [475, 500].



def convergence_check(measurements, rescale, search_rank, regularization, tolerance, step, trials, num_measurements=500, size=50, rank=5, default_rescale=1, nesterov=False, momentum=1, max_count=10000, stochastic=False, batch_size=10, grace=0, Thres_1=0, Thres_2=0, Thres_3=0, alt_reg_1=0, alt_reg_2=0, alt_reg_3=0):
    measurements = measurements
    ground_truth_mat = ground_truth(size, rank, symmetric=True)

    # Run optimization
    results = []
    success = 0
    error_document = np.zeros((trials, max_count+1))
    for _ in range(trials):
        X_init = initialization_X(size, search_rank, rescale=rescale)
        error, optimized_X, count, error_doc = optimization_BM_loss_altreg(            
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
            grace=grace,
            threshold_1=Thres_1,
            threshold_2=Thres_2,
            threshold_3=Thres_3,
            alt_regularization_1=alt_reg_1,
            alt_regularization_2=alt_reg_2,
            alt_regularization_3=alt_reg_3)
        
        results.append((error, optimized_X, count))
        error_document[_] = error_doc
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

    error_document = np.mean(error_document, axis=0)


    return success, error, optimized_X, count, error_document

    
### The following function will attempt to find the optimal sample size for a !given search rank and regularization parameter.
def sample_size_experiment(measurements, default_rescale, search_rank, regularization, tolerance, step, trials, lb, ub, jump=100, size=50, rank=5, nesterov=False, momentum=1, max_count=10000, stochastic=False, batch_size=10, grace=0, Thres_1=0, Thres_2=0, Thres_3=0, alt_reg_1=0, alt_reg_2=0, alt_reg_3=0):
    num_measurements_list = measurements
    
    for num_measurements in num_measurements_list:
        success, error, optimized_X, count, error_document = convergence_check(num_measurements, default_rescale, search_rank=search_rank, regularization=regularization, tolerance=tolerance, step=step, trials=trials, num_measurements=num_measurements, size=size, rank=rank, nesterov=nesterov, momentum=momentum, max_count=max_count, stochastic=stochastic, batch_size=batch_size, grace=grace, Thres_1=Thres_1, Thres_2=Thres_2, Thres_3=Thres_3, alt_reg_1=alt_reg_1, alt_reg_2=alt_reg_2, alt_reg_3=alt_reg_3)
        actual_search_rank = optimized_X.shape[1]
        if success:
            print(f"Search rank {actual_search_rank} at sample size {num_measurements} succeeded with error {error}, iterations {count}")
            return num_measurements, error, count, error_document
        print(f"Search rank {actual_search_rank} failed at sample size {num_measurements} with error {error}, iterations {count}")
    print(f"Search rank {actual_search_rank} No successful sample size found in the given range.")
    return 0, error, count, error_document

# Fixed parameters: reasonable tolerance and step size
# sample_size_experiment(5, 0, 0.6, 0.5, 7, 500, 700, jump=100, size=50, rank=5, default_rescale=1, nesterov=False, momentum=1, max_count=100)