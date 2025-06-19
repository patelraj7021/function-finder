import numpy as np

def quadratic(x):
    return x**2

# Generate dummy data with noise: y = x^2 + noise
def generate_dummy_data(func, n_samples=1000, noise_std=0.1, x_range=(-5, 5)):
    # Generate x values uniformly distributed in the given range
    x = np.linspace(x_range[0], x_range[1], n_samples)
    
    # Generate true y values
    y_true = func(x)
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, n_samples)
    y = y_true + noise
    
    return x, y, y_true