import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import time
from Initialization import initialization_measurements_Ginibre, ground_truth, initialization_X
from Optimization import optimization_BM_loss
from Testing import sample_size_experiment

###!!!!!!!!! alter the function in Optimization.py to turn on or off the l1 mimic gradient !!!!!!!!!!!###


def process_single_search_rank(args):
    """
    Wrapper function for multiprocessing that processes a single search rank.
    """
    default_rescale, search_rank, regularization, tolerance, max_iterations, step, trials, lb, ub, jump, size, rank, rescale, nesterov, momentum, stochastic, Thres_1, Thres_2, Thres_3, alt_reg_1, alt_reg_2, alt_reg_3 = args

    print(f"Starting search rank {search_rank}...")
    start_time = time.time()
    
    temp_complexity, temp_error, temp_count = sample_size_experiment(default_rescale,
        search_rank, regularization, tolerance=tolerance, max_count=max_iterations, step=step, trials=trials, 
        lb=lb, ub=ub, jump=jump, size=size, rank=rank, nesterov=nesterov, momentum=momentum, Thres_1=Thres_1, Thres_2=Thres_2, Thres_3=Thres_3, alt_reg_1=alt_reg_1, alt_reg_2=alt_reg_2, alt_reg_3=alt_reg_3
    )

    
    end_time = time.time()
    print(f"Completed search rank {search_rank} in {end_time - start_time:.2f} seconds - Sample complexity: {temp_complexity}")
    
    return [search_rank, temp_complexity, temp_error, temp_count]

if __name__ == "__main__":
    # Configuration
    size = 40
    rank = 5
    default_rescale = 0.000000000001
    default_regularization = 0
    default_search_rank = 30
    lower_bound_complexity = 500
    upper_bound_complexity = 600
    jump_size = 50
    max_iterations = 10000
    default_search_rank_ub= 31
    default_step = 1
    nesterov = True
    momentum = 0.001
    tolerance = 0.01
    stochastic = False  # Set to True if you want to use stochastic optimization
    batch_size = 50  # Batch size for stochastic optimization. It's time to make a choice between speed and accuracy.
    Thres_1=0.1
    Thres_2=0
    Thres_3=0
    alt_reg_1=0
    alt_reg_2=0
    alt_reg_3=0
    # Choose between sequential and parallel processing
    USE_MULTIPROCESSING = True
    NUM_PROCESSES = min(cpu_count(), 20)  # Use up to 20 processes or number of CPU cores, whichever is smaller

    print(f"Using {'multiprocessing' if USE_MULTIPROCESSING else 'sequential processing'}")
    if USE_MULTIPROCESSING:
        print(f"Number of processes: {NUM_PROCESSES}")
    
    ### For simplicity we first set the regularization to 0, and the search rank to 5. In the following experiments we will increase the search rank gradually from 5 to 20.###
    sample_complexity = []
    
    if USE_MULTIPROCESSING:
        # Prepare arguments for multiprocessing
        search_ranks = list(range(default_search_rank, default_search_rank_ub, 1))
        args_list = [
            (default_rescale, search_rank, default_regularization, tolerance, max_iterations, default_step, 5, 
             lower_bound_complexity, upper_bound_complexity, jump_size, 
             size, rank, default_rescale, nesterov, momentum, stochastic, Thres_1, Thres_2, Thres_3, alt_reg_1, alt_reg_2, alt_reg_3)
            for search_rank in search_ranks
        ]
        
        # Use multiprocessing
        start_time = time.time()
        try:
            with Pool(processes=NUM_PROCESSES) as pool:
                results = pool.map(process_single_search_rank, args_list)
        except Exception as e:
            print(f"Multiprocessing failed: {e}")
            print("Falling back to sequential processing...")
            USE_MULTIPROCESSING = False
            
        if USE_MULTIPROCESSING:
            end_time = time.time()
            
            # Filter out failed results (complexity = 0) and sort by search rank
            sample_complexity = [result for result in results if result[1] != 0]
            sample_complexity.sort(key=lambda x: x[0])  # Sort by search rank
            
            print(f"\nMultiprocessing completed in {end_time - start_time:.2f} seconds")
        
    if not USE_MULTIPROCESSING:
        # Sequential processing (fallback)
        print("Running sequential processing...")
        temp_complexity = lower_bound_complexity
        for search_rank in range(default_search_rank, default_search_rank_ub, 1):
            temp_complexity, temp_error, temp_count = sample_size_experiment(
                default_rescale, search_rank, default_regularization, tolerance=tolerance, step=default_step, trials=5, 
                lb=temp_complexity, ub=upper_bound_complexity, jump=jump_size, 
                size=size, rank=rank, stochastic=stochastic,
                nesterov=nesterov, momentum=momentum, max_count=max_iterations, Thres_1=Thres_1, Thres_2=Thres_2, Thres_3=Thres_3, alt_reg_1=alt_reg_1, alt_reg_2=alt_reg_2, alt_reg_3=alt_reg_3
            )
            sample_complexity.append([search_rank, temp_complexity, temp_error, temp_count])
            print(f"Search rank {search_rank}, Sample complexity: {temp_complexity}")
            if temp_complexity == 0:
                print(f"Search rank bigger {search_rank} will not be converging in the given range, experiments skipped.")        
                break
    
    print("Sample complexity for search ranks from 5 to 20:")
    for rank, complexity, error, count in sample_complexity:
        print(f"Search rank {rank}: Sample complexity {complexity}, Error {error}, Iterations {count}")

    # Extract data for plotting
    errors = [data[2] for data in sample_complexity]
    search_ranks = [data[0] for data in sample_complexity]
    complexities = [data[1] for data in sample_complexity]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(search_ranks, errors, 'b-o', linewidth=2, markersize=8, label='error')
    plt.xlabel('Search Rank', fontsize=12)
    plt.ylabel('error', fontsize=12)
    plt.title('Relationship between Search Rank and error', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add value labels on each point
    for i, (rank, err) in enumerate(zip(search_ranks, errors)):
        plt.annotate(f'{err}', (rank, errors[i]), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()

    # Save the plot
    plt.savefig('search_rank_vs_final_error.png', dpi=300, bbox_inches='tight')
    print("Graph saved as 'search_rank_vs_final_error.png'")
