import matplotlib.pyplot as plt
import numpy as np

# Read epoch error data
try:
    data = np.loadtxt('epoch_error.txt')
    epochs = data[:, 0]
    errors = data[:, 1]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, errors, 'b-', linewidth=2, label='Epoch Error')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title('Training Error Per Epoch')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')  # Log scale for better visualization
    
    # Show statistics
    print(f"Initial Error: {errors[0]:.6f}")
    print(f"Final Error: {errors[-1]:.6f}")
    print(f"Error Reduction: {((errors[0] - errors[-1]) / errors[0] * 100):.2f}%")
    
    plt.tight_layout()
    plt.savefig('epoch_error_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
except FileNotFoundError:
    print("Error: epoch_error.txt not found. Run the training first.")
except Exception as e:
    print(f"Error plotting: {e}")