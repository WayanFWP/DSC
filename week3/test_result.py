import pandas as pd
import matplotlib.pyplot as plt

# Create a figure with multiple subplots
plt.figure(figsize=(15, 10))

# Plot 1: Training errors per iteration
try:
    iter_df = pd.read_csv('training_errors_per_iteration.csv')
    plt.subplot(2, 3, 1)
    plt.plot(iter_df['Iteration'], iter_df['Training_Error'], linewidth=0.5, alpha=0.7)
    plt.title('Training Error Per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Training Error')
    plt.grid(True, alpha=0.3)
except Exception as e:
    print(f"Could not plot iteration errors: {e}")

# Plot 2: Training errors per epoch
try:
    epoch_df = pd.read_csv('training_errors_per_epoch.csv')
    plt.subplot(2, 3, 2)
    plt.plot(epoch_df['Epoch'], epoch_df['Training_Error'], linewidth=2)
    plt.title('Training Error Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Training Error')
    plt.grid(True, alpha=0.3)
except Exception as e:
    print(f"Could not plot epoch errors: {e}")

# Plot 3: Test detection errors
try:
    test_df = pd.read_csv('test_results.csv')
    plt.subplot(2, 3, 3)
    plt.bar(range(len(test_df)), test_df['Detection_Error'])
    plt.title('Test Detection Errors')
    plt.xlabel('Test Case')
    plt.ylabel('Detection Error')
    plt.xticks(range(len(test_df)), test_df['Filename'], rotation=45)
except Exception as e:
    print(f"Could not plot test errors: {e}")

# Plot 4: User input errors (if exists)
try:
    user_df = pd.read_csv('user_input_results.csv')
    plt.subplot(2, 3, 4)
    plt.bar(range(len(user_df)), user_df['Detection_Error'])
    plt.title('User Input Detection Errors')
    plt.xlabel('Input Number')
    plt.ylabel('Detection Error')
except Exception as e:
    print(f"Could not plot user errors: {e}")

# Plot 5: Confidence vs Detection Error (Test data)
try:
    test_df = pd.read_csv('test_results.csv')
    plt.subplot(2, 3, 5)
    plt.scatter(test_df['Confidence'], test_df['Detection_Error'], alpha=0.7)
    plt.title('Confidence vs Detection Error')
    plt.xlabel('Confidence (%)')
    plt.ylabel('Detection Error')
    plt.grid(True, alpha=0.3)
except Exception as e:
    print(f"Could not plot confidence scatter: {e}")

# Plot 6: Training error comparison (last 1000 iterations vs epochs)
try:
    iter_df = pd.read_csv('training_errors_per_iteration.csv')
    epoch_df = pd.read_csv('training_errors_per_epoch.csv')
    
    plt.subplot(2, 3, 6)
    # Plot last 1000 iterations if available
    if len(iter_df) > 1000:
        last_1000 = iter_df.tail(1000)
        plt.plot(last_1000['Iteration'], last_1000['Training_Error'], 
                label='Per Iteration (Last 1000)', alpha=0.7, linewidth=0.5)
    else:
        plt.plot(iter_df['Iteration'], iter_df['Training_Error'], 
                label='Per Iteration', alpha=0.7, linewidth=0.5)
    
    # Plot epochs on secondary axis
    ax2 = plt.gca().twinx()
    ax2.plot(epoch_df['Epoch'] * len(iter_df) // len(epoch_df), epoch_df['Training_Error'], 
            'r-', label='Per Epoch', linewidth=2)
    ax2.set_ylabel('Epoch Error', color='r')
    
    plt.title('Training Error Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Iteration Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
except Exception as e:
    print(f"Could not plot comparison: {e}")

plt.tight_layout()
plt.show()

# Print summary statistics
try:
    iter_df = pd.read_csv('training_errors_per_iteration.csv')
    print(f"\n📊 Training Statistics:")
    print(f"Total iterations: {len(iter_df)}")
    print(f"Initial error: {iter_df['Training_Error'].iloc[0]:.6f}")
    print(f"Final error: {iter_df['Training_Error'].iloc[-1]:.6f}")
    print(f"Min error: {iter_df['Training_Error'].min():.6f}")
    print(f"Max error: {iter_df['Training_Error'].max():.6f}")
    print(f"Error reduction: {((iter_df['Training_Error'].iloc[0] - iter_df['Training_Error'].iloc[-1]) / iter_df['Training_Error'].iloc[0] * 100):.2f}%")
except Exception as e:
    print(f"Could not print statistics: {e}")