import numpy as np
import matplotlib.pyplot as plt
from Initialization import initialization_measurements_Ginibre, ground_truth, initialization_X
from Optimization import optimization_BM_loss
from Testing import sample_size_experiment

if __name__ == "__main__":
    # Configuration
    num_measurements = 500
    size = 50
    rank = 5
    default_rescale = 1
    default_regularization = 0
    default_search_rank = 5

### For simplicity we first set the regularization to 0, and the search rank to 5. In the following experiments we will increase the search rank gradually from 5 to 20.###
sample_complexity=[]
for search_rank in range(5, 21, 1):
    temp_complexity=sample_size_experiment(search_rank, default_regularization, tolerance=0.6, step=0.5, trials=5, lb=500, ub=2500, jump=100, size=50, rank=5, default_rescale=1)
    sample_complexity.append([search_rank, temp_complexity])
    print(f"Search rank {search_rank}, Sample complexity: {temp_complexity}")

print("Sample complexity for search ranks from 5 to 20:")
for rank, complexity in sample_complexity:
    print(f"Search rank {rank}: Sample complexity {complexity}")

# Extract data for plotting
search_ranks = [data[0] for data in sample_complexity]
complexities = [data[1] for data in sample_complexity]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(search_ranks, complexities, 'b-o', linewidth=2, markersize=8, label='Sample Complexity')
plt.xlabel('Search Rank', fontsize=12)
plt.ylabel('Sample Complexity', fontsize=12)
plt.title('Relationship between Search Rank and Sample Complexity', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# Add value labels on each point
for i, (rank, complexity) in enumerate(zip(search_ranks, complexities)):
    plt.annotate(f'{complexity}', (rank, complexity), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9)

plt.tight_layout()
plt.show()

# Save the plot
plt.savefig('search_rank_vs_sample_complexity.png', dpi=300, bbox_inches='tight')
print("Graph saved as 'search_rank_vs_sample_complexity.png'")
