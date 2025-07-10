import matplotlib.pyplot as plt
import numpy as np

# Read iteration error data
try:
    data = np.loadtxt('iteration_error.txt')
    iterations = data[:, 0]
    errors = data[:, 1]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(iterations, errors, 'b-', linewidth=1, alpha=0.7, label='Iteration Error')
    
    # Add moving average for smoother visualization
    window_size = max(1, len(errors) // 100)  # 1% of total iterations
    if len(errors) > window_size:
        moving_avg = np.convolve(errors, np.ones(window_size)/window_size, mode='valid')
        moving_avg_x = iterations[window_size-1:]
        plt.plot(moving_avg_x, moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window_size})')
    
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.title('Training Error Per Iteration')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')  # Log scale for better visualization
    
    # Show statistics
    print(f"Total iterations: {len(iterations)}")
    print(f"Initial Error: {errors[0]:.6f}")
    print(f"Final Error: {errors[-1]:.6f}")
    print(f"Error Reduction: {((errors[0] - errors[-1]) / errors[0] * 100):.2f}%")
    print(f"Min Error: {np.min(errors):.6f}")
    print(f"Max Error: {np.max(errors):.6f}")
    
    plt.tight_layout()
    plt.savefig('iteration_error_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
except FileNotFoundError:
    print("Error: iteration_error.txt not found. Run the training first.")
except Exception as e:
    print(f"Error plotting: {e}")