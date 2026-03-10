import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from Initialization import initialization_measurements_Ginibre, ground_truth, initialization_X, initialization_measurements_prGaus
from Optimization import optimization_BM_loss
from Testing import convergence_check, sample_size_experiment
from Gradients_and_Heissian import get_gradient_BM_loss, get_gradient_convex_loss, get_gradient_BM_loss_l1mimic, adaptive_regularization_proto, Constant_regularization

###!!!!!!!!! alter the function in Optimization.py to turn on or off the l1 mimic gradient !!!!!!!!!!!###

#The measurements is a list of measurement of different sample sizes, which will be shared across different search ranks. We will use the Ginibre ensemble for initialization, which is a common choice in phase retrieval problems.

def phase_diagram_single_rank(measurements, rescale, search_rank, regularization, tolerance, step, trials, num_measurements=500, size=50, rank=5, nesterov=False, momentum=1, max_count=10000, stochastic=False, batch_size=10, grace=0, reg_func=None, rounds=1):
    print(f"Starting search rank {search_rank}...")
    overall_performance=np.zeros((rounds, len(measurements)))
    #in each round we will run the optimization process from the lowest sample size to the highest sample size, each recording the fraction of successful trials.
    for i in range(rounds):
        for j in range(len(measurements)):
            success, error, optimized_X, count, error_document = convergence_check(measurements[j], rescale, search_rank=search_rank, regularization=regularization, tolerance=tolerance, step=step, trials=trials, num_measurements=num_measurements, size=size, rank=rank, nesterov=nesterov, momentum=momentum, max_count=max_count, stochastic=stochastic, batch_size=batch_size, grace=grace, reg_func=reg_func)
            overall_performance[i, j] = success
    #For each complexity level we take the average success rate across rounds, and print the results.
    average_performance = np.mean(overall_performance, axis=0)
    print(f"Completed search rank {search_rank} - Average success rate: {average_performance}")
    return average_performance


def process_search_rank_wrapper(search_rank, measurements, rescale, regularization, tolerance, step, trials, num_measurements, size, rank, nesterov, momentum, max_count, stochastic, batch_size, grace, reg_func, rounds):
    """Wrapper function for multiprocessing - processes a single search rank"""
    return phase_diagram_single_rank(
        measurements=measurements,
        rescale=rescale,
        search_rank=search_rank,
        regularization=regularization,
        tolerance=tolerance,
        step=step,
        trials=trials,
        num_measurements=num_measurements,
        size=size,
        rank=rank,
        nesterov=nesterov,
        momentum=momentum,
        max_count=max_count,
        stochastic=stochastic,
        batch_size=batch_size,
        grace=grace,
        reg_func=reg_func,
        rounds=rounds
    )


#1. for each search rank, we will run the optimization process from the lowest sample size to the highest sample size, each recording the fraction of successful trials.
if __name__ == "__main__":
    # Configuration
    size = 50
    rank = 1
    default_rescale = 1
    default_regularization = 10000
    default_search_rank = 1
    lower_bound_complexity = 50
    upper_bound_complexity = 300
    jump_size = 10
    max_iterations = 5000
    default_search_rank_ub =20
    default_step = 0.8
    trials = 1
    nesterov = True
    momentum = 0
    tolerance = 0.01
    stochastic = True # Set to True if you want to use stochastic optimization
    batch_size = 40  # Batch size for stochastic optimization
    reg_func = adaptive_regularization_proto   # Set to None if you want to use fixed regularization, or provide a custom function for adaptive regularization.
    rounds = 10 # Number of rounds to average the results over. Set to 1 for a single run, or higher for more robust statistics.

    # Generate measurements that will be shared across all search ranks
    measurements = [initialization_measurements_Ginibre(i, size, rescale=False) for i in range(lower_bound_complexity, upper_bound_complexity+jump_size, jump_size)]
    
    # Define the range of search ranks to process
    search_rank_range = list(range(default_search_rank, default_search_rank_ub + 1))
    
    # Determine optimal number of processes (don't exceed number of search ranks)
    num_processes = min(cpu_count(), len(search_rank_range))
    
    print(f"Starting multiprocessing with {num_processes} processes for {len(search_rank_range)} search rank(s)...")
    start_time = time.time()
    
    # Use multiprocessing Pool to parallelize search rank processing
    with Pool(processes=num_processes) as pool:
        # Create a partial function with fixed parameters
        process_func = partial(
            process_search_rank_wrapper,
            measurements=measurements,
            rescale=default_rescale,
            regularization=default_regularization,
            tolerance=tolerance,
            step=default_step,
            trials=trials,
            num_measurements=measurements[0].shape[0],
            size=size,
            rank=rank,
            nesterov=nesterov,
            momentum=momentum,
            max_count=max_iterations,
            stochastic=stochastic,
            batch_size=batch_size,
            grace=0,
            reg_func=reg_func,
            rounds=rounds
        )
        
        # Map the function to each search rank (each processor handles one rank)
        performance_across_ranks = pool.map(process_func, search_rank_range)
    
    elapsed_time = time.time() - start_time
    print(f"Multiprocessing completed in {elapsed_time:.2f} seconds")

    # Now we draw the phase diagram, with complexity on the y-axis, search rank on the x-axis, and color representing the fraction of success.
    # Prepare data for phase diagram
    sample_complexities = np.arange(lower_bound_complexity, upper_bound_complexity + jump_size, jump_size)
    search_ranks = np.arange(default_search_rank, default_search_rank_ub + 1)

    # Create a 2D array for the phase diagram
    # Rows correspond to sample complexities, columns correspond to search ranks
    phase_data = np.zeros((len(sample_complexities), len(search_ranks)))

    for j, performance in enumerate(performance_across_ranks):
        if performance is not None:
            phase_data[:, j] = performance

    # Create the phase diagram
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(phase_data, aspect='auto', cmap='RdYlGn', origin='lower', 
                   extent=[search_ranks[0]-0.5, search_ranks[-1]+0.5, sample_complexities[0]-jump_size/2, sample_complexities[-1]+jump_size/2],
                   vmin=0, vmax=1)

    # Set ticks and labels for axes
    ax.set_xticks(search_ranks)
    ax.set_yticks(sample_complexities)
    ax.set_xlabel('Searching Rank', fontsize=12)
    ax.set_ylabel('Sample Complexity', fontsize=12)
    ax.set_title('Phase Diagram: Success Rate vs Rank and Sample Complexity', fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fraction of Success', fontsize=12)

    plt.tight_layout()
    plt.savefig('phase_diagram.png', dpi=300)
    plt.show()

