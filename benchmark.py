import time
import numpy as np
import subprocess
import re
import sys
import matplotlib.pyplot as plt
from SklearnNeuralNetwork import SklearnNeuralNetwork

# --- Benchmark Functions ---

def run_cpp_benchmark(num_samples):
    """Runs the C++ benchmark executable and parses the output for total time."""
    try:
        result = subprocess.run(
            ['./benchmark_cpp', str(num_samples)], 
            capture_output=True, 
            text=True, 
            check=True
        )
        output = result.stdout
        
        # Parse "Total time: X seconds"
        match = re.search(r"Total time: ([0-9.]+) seconds", output)
        if match:
            return float(match.group(1))
        else:
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error running C++ benchmark: {e}")
        return None

def run_sklearn_benchmark(num_samples):
    """Runs the Sklearn benchmark."""
    input_size = 784
    output_size = 10
    
    # Generate random data
    X = np.random.rand(num_samples, input_size)
    y = np.random.randint(0, output_size, num_samples)

    nn = SklearnNeuralNetwork()
    
    # Initialize
    init_samples = min(100, num_samples)
    nn.train_dummy(X[:init_samples], y[:init_samples])
    
    start_time = time.time()
    _ = nn.predict(X)
    end_time = time.time()
    
    return end_time - start_time

# --- Main Logic ---

def run_comparison():
    sample_sizes = [10, 100, 1000, 5000, 10000, 20000, 50000, 100000]
    
    sklearn_times = []
    cpp_times = []
    valid_sizes = []

    print(f"{'Samples':<10} | {'Sklearn (s)':<12} | {'C++ (s)':<12} | {'Speedup':<20}")
    print("-" * 60)

    for n in sample_sizes:
        print(f"Running for n={n}...", end='\r')
        
        t_sklearn = run_sklearn_benchmark(n)
        t_cpp = run_cpp_benchmark(n)
        
        if t_cpp is not None:
            speedup = t_sklearn / t_cpp
            print(f"{n:<10} | {t_sklearn:<12.6f} | {t_cpp:<12.6f} | {speedup:<20.2f}")
            
            sklearn_times.append(t_sklearn)
            cpp_times.append(t_cpp)
            valid_sizes.append(n)
        else:
            print(f"{n:<10} | {t_sklearn:<12.6f} | {'ERROR':<12} | {'-':<20}")

    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Time vs Samples (Log Scale)
    plt.subplot(1, 2, 1)
    plt.plot(valid_sizes, sklearn_times, 'o-', label='Sklearn (Python)')
    plt.plot(valid_sizes, cpp_times, 's-', label='Custom C++')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Samples')
    plt.ylabel('Inference Time (seconds)')
    plt.title('Inference Time Comparison (Log-Log)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)

    # Plot 2: Speedup vs Samples
    plt.subplot(1, 2, 2)
    speedups = [s / c for s, c in zip(sklearn_times, cpp_times)]
    plt.plot(valid_sizes, speedups, '^-', color='green')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Parity')
    plt.xscale('log')
    plt.xlabel('Number of Samples')
    plt.ylabel('Speedup (C++ / Sklearn)')
    plt.title('C++ Speedup Factor')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    print("\nResults plotted to 'benchmark_results.png'")

if __name__ == "__main__":
    run_comparison()
