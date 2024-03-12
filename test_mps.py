import torch
import time

times = []


def benchmark(device):
    # Create a large tensor
    x = torch.randn(20000, 20000, device=device)

    # Start the timer
    start_time = time.time()

    # Perform a simple operation (e.g., matrix multiplication)
    result = x @ x

    # Stop the timer
    end_time = time.time()

    # Calculate and print the duration
    print(f"Device: {device}, Time taken: {end_time - start_time:.4f} seconds")

    times.append(end_time - start_time)


# Check if MPS is supported and run the benchmark
if torch.backends.mps.is_available():
    # Run on MPS
    benchmark("mps")
else:
    print("MPS not available on this system.")

# Run on CPU
benchmark("cpu")

# Calculate the speedup
speedup = times[1] / times[0]
print(f"Speedup: {speedup:.2f}x")