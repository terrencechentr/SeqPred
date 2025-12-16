import numpy as np
from typing import Union

def calculate_stats(data: Union[np.ndarray, list]) -> tuple[float, float]:
    """
    Calculate mean and standard deviation of a list or numpy array
    
    Args:
        data: List or numpy array of numeric values
        
    Returns:
        tuple: (mean, std) as floats
    """
    # Convert to numpy array if it's a list
    if isinstance(data, list):
        data = np.array(data)
    
    # Calculate mean and standard deviation
    mean = np.mean(data)
    std = np.std(data)
    
    return mean, std

def change_rate_to_data(change_rates: Union[np.ndarray, list], initial_value: float = 1.0) -> np.ndarray:
    """
    Convert change rates to actual data values
    
    Args:
        change_rates: List or numpy array of change rates (as percentages)
        initial_value: Starting value for the data series
        
    Returns:
        numpy array: Actual data values
    """
    # Convert to numpy array if it's a list
    if isinstance(change_rates, list):
        change_rates = np.array(change_rates)
    
    # Convert percentage change rates to multipliers (e.g., 2.5% -> 1.025)
    multipliers = 1 + change_rates / 100
    
    # Calculate cumulative product to get the data values
    data = initial_value * np.cumprod(multipliers)
    
    return data


def data_to_change_rate(data: Union[np.ndarray, list]) -> np.ndarray:
    """
    Convert actual data values to change rates
    
    Args:
        data: List or numpy array of data values
        
    Returns:
        numpy array: Change rates as percentages
    """
    # Convert to numpy array if it's a list
    if isinstance(data, list):
        data = np.array(data)

    change_present = (data[1:] - data[:-1]) / data[:-1]
    
    # Calculate change rates as percentages
    return change_present * 100


def random_sequence_sample(data, seq_len, seq_num=None):
    """
    Generate random sequence samples from a list of data
    
    Args:
        data: List or numpy array of data values
        seq_len: Length of each sequence
        seq_num: Number of sequences to generate (if None, return all possible sequences)
        
    Returns:
        list: List of sequences, each sequence is a list of seq_len elements
    """
    data = np.array(data) if isinstance(data, list) else data
    
    # Calculate maximum possible sequences
    max_start_idx = len(data) - seq_len
    if max_start_idx < 0:
        raise ValueError(f"Data length ({len(data)}) is less than sequence length ({seq_len})")
    
    max_possible_sequences = max_start_idx + 1
    
    # If seq_num is None  return all possible sequences
    if seq_num is None:
        sequences = []
        for start_idx in range(max_possible_sequences):
            sequence = data[start_idx:start_idx + seq_len].tolist()
            sequences.append(sequence)
        return sequences
    
    # Otherwise, generate random sequences as before
    sequences = []
    for _ in range(seq_num):
        start_idx = np.random.randint(0, max_start_idx + 1)
        sequence = data[start_idx:start_idx + seq_len].tolist()
        sequences.append(sequence)
    
    sequences = [np.array(seq) for seq in sequences]

    return sequences


def add_noise_to_array(array, noise_mean=0, noise_std=0.1):
    """
    Add noise to an array.
    
    Args:
        array: Input array to add noise to
        noise_std: Standard deviation of the Gaussian noise (default: 0.1)
        noise_mean: Mean of the noise (default: 0)
    Returns:
        Array with added noise
    """
    
    # Generate noise with the same shape as the input array
    noise = np.random.normal(noise_mean, noise_std, array.shape)
    
    # Add noise to the original array
    noisy_array = array + noise
    
    return noisy_array