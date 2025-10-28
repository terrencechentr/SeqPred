import matplotlib.pyplot as plt
import numpy as np

def visualize_list(data, title="List Visualization", xlabel="Index", ylabel="Values", color='blue'):
    """
    Visualize a single list of values using matplotlib
    
    Args:
        data (list): List of values to visualize
        title (str): Title for the plot
        xlabel (str): Label for x-axis
        ylabel (str): Label for y-axis
        color (str): Color for the plot line
    """
    plt.figure(figsize=(12, 6))
    
    # Create indices for plotting
    indices = range(len(data))
    
    # Plot the list
    plt.plot(indices, data, color=color, linewidth=2, alpha=0.8)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print basic statistics
    print(f"Data Statistics:")
    print(f"Length: {len(data)}")
    print(f"Mean: {np.mean(data):.4f}")
    print(f"Standard Deviation: {np.std(data):.4f}")
    print(f"Min: {np.min(data):.4f}")
    print(f"Max: {np.max(data):.4f}")

    return plt.gcf()


def visualize_two_lists(true, pred, title="Comparison of Two Lists", label1="True", label2="Prediction"):
    """
    Visualize two lists of floats using matplotlib
    
    Args:
        list1 (list): First list of float values
        list2 (list): Second list of float values  
        title (str): Title for the plot
        label1 (str): Label for the first list
        label2 (str): Label for the second list
    """
    plt.figure(figsize=(15, 8))
    
    # Create time indices for plotting
    time_indices = range(len(true))
    
    # Plot both lists
    plt.plot(time_indices, true, label=label1, color='blue', alpha=0.7, linewidth=2)
    plt.plot(time_indices, pred, label=label2, color='red', alpha=0.7, linewidth=2)
    
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print some statistics
    print(f"{label1} mean: {np.mean(true):.4f}")
    print(f"{label2} mean: {np.mean(pred):.4f}")
    print(f"{label1} std: {np.std(true):.4f}")
    print(f"{label2} std: {np.std(pred):.4f}")
    print(f"Correlation: {np.corrcoef(true, pred)[0,1]:.4f}")
    print(f"Mean Absolute Difference: {np.mean(np.abs(np.array(true) - np.array(pred))):.4f}")

    return plt.gcf()
