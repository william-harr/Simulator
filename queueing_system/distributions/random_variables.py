"""
Random variable generators for queueing systems.
Compatible with scipy.stats distributions and custom generators.
"""

import numpy as np
import random as rand
from typing import Callable, Optional, List
from scipy import stats


# Basic distributions
def exponential(rate: float) -> float:
    """Generate exponential random variable."""
    if rate <= 0:
        return 0.0
    return -np.log(np.random.random()) / rate


def uniform(a: float, b: float) -> float:
    """Generate uniform random variable between a and b."""
    return a + (b - a) * np.random.random()


def normal(mean: float, std: float) -> float:
    """Generate normal random variable."""
    return np.random.normal(mean, std)


def gamma(shape: float, scale: float) -> float:
    """Generate gamma random variable."""
    return np.random.gamma(shape, scale)


def weibull(shape: float, scale: float) -> float:
    """Generate Weibull random variable."""
    return scale * np.random.weibull(shape)


def triangular(a: float, b: float, c: float) -> float:
    """Generate triangular random variable."""
    return np.random.triangular(a, c, b)


def project_rv() -> float:
    """Your original project random variable."""
    U = rand.random()
    X = 2 * np.sqrt(U)
    return X


def lognormal(mean: float, std: float) -> float:
    """Generate log-normal random variable."""
    return np.random.lognormal(mean, std)


def beta(a: float, b: float) -> float:
    """Generate beta random variable."""
    return np.random.beta(a, b)


def pearson3(shape: float, scale: float, loc: float = 0) -> float:
    """Generate Pearson Type III (shifted gamma) random variable."""
    # Pearson Type III is essentially a shifted gamma distribution
    return stats.pearson3.rvs(shape, loc=loc, scale=scale)
    
def pearson6(beta: float, p: float, q: float, loc: float = 0) -> float:
    """Generate Pearson Type VI (Beta Prime) random variable."""
    # Pearson Type VI is also known as Beta Prime distribution
    # It's related to the F-distribution
    return stats.betaprime.rvs(p, q, scale=beta) + loc


# Distribution factory functions
def lognormal_distribution(mean: float, std: float) -> Callable[[], float]:
    """Create a log-normal distribution function."""
    return lambda: lognormal(mean, std)


def beta_distribution(a: float, b: float) -> Callable[[], float]:
    """Create a beta distribution function."""
    return lambda: beta(a, b)


def pearson3_distribution(shape: float, scale: float, loc: float = 0) -> Callable[[], float]:
    """Create a Pearson Type III distribution function."""
    return lambda: pearson3(shape, scale, loc)


def exponential_distribution(rate: float) -> Callable[[], float]:
    """Create an exponential distribution function."""
    return lambda: exponential(rate)


def uniform_distribution(a: float, b: float) -> Callable[[], float]:
    """Create a uniform distribution function."""
    return lambda: uniform(a, b)


def normal_distribution(mean: float, std: float) -> Callable[[], float]:
    """Create a normal distribution function."""
    return lambda: normal(mean, std)


def gamma_distribution(shape: float, scale: float) -> Callable[[], float]:
    """Create a gamma distribution function."""
    return lambda: gamma(shape, scale)

def pearson6_distribution(beta: float, p: float, q: float, loc: float = 0) -> Callable[[], float]:
    """Create a Pearson Type VI distribution function."""
    return lambda: pearson6(beta, p, q, loc)

def deterministic_distribution(value: float) -> Callable[[], float]:
    """Create a deterministic distribution (always returns same value)."""
    return lambda: value


def empirical_distribution(data: List[float], replace: bool = True) -> Callable[[], float]:
    """Create an empirical distribution from data."""
    data_array = np.array(data)
    return lambda: np.random.choice(data_array, replace=replace)
    
def weibull_distribution(shape: float, scale: float) -> Callable[[], float]:
    """Create a Weibull distribution function."""
    return lambda: weibull(shape, scale)


# Advanced distributions using scipy
def scipy_distribution(dist_name: str, **params) -> Callable[[], float]:
    """
    Create a distribution function from scipy.stats.
    
    Examples:
        scipy_distribution('norm', loc=10, scale=2)  # Normal(10, 2)
        scipy_distribution('gamma', a=2, scale=1/3)  # Gamma(2, 1/3)
        scipy_distribution('beta', a=2, b=5)         # Beta(2, 5)
    """
    dist = getattr(stats, dist_name)
    return lambda: dist.rvs(**params)


# Composite distributions
def mixture_distribution(distributions: List[Callable[[], float]],
                        weights: Optional[List[float]] = None) -> Callable[[], float]:
    """
    Create a mixture distribution from multiple distributions.
    
    Args:
        distributions: List of distribution functions
        weights: Probability weights for each distribution (must sum to 1)
    """
    if weights is None:
        weights = [1/len(distributions)] * len(distributions)
    
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize
    
    def sample():
        dist_idx = np.random.choice(len(distributions), p=weights)
        return distributions[dist_idx]()
    
    return sample


def phase_type_distribution(rates: List[float],
                           initial_probs: Optional[List[float]] = None) -> Callable[[], float]:
    """
    Create a phase-type distribution (sum of exponentials).
    
    Args:
        rates: List of rates for each phase
        initial_probs: Initial probabilities for each phase
    """
    if initial_probs is None:
        initial_probs = [1.0] + [0.0] * (len(rates) - 1)
    
    def sample():
        total_time = 0
        current_phase = np.random.choice(len(rates), p=initial_probs)
        
        while current_phase < len(rates):
            # Time in current phase
            phase_time = exponential(rates[current_phase])
            total_time += phase_time
            
            # Move to next phase or exit
            if np.random.random() < 0.5:  # Exit probability
                break
            current_phase += 1
        
        return total_time
    
    return sample


# Customer-dependent distributions
def order_size_dependent_distribution(base_rate: float) -> Callable[[object], float]:
    """
    Create a distribution that depends on customer order size.
    
    Args:
        base_rate: Base processing rate
    
    Returns:
        Function that takes a customer and returns processing time
    """
    def sample(customer):
        # Processing time increases with order size
        adjusted_rate = base_rate / customer.order_size
        return exponential(adjusted_rate)
    
    return sample


def priority_dependent_distribution(base_mean: float) -> Callable[[object], float]:
    """
    Create a distribution that depends on customer priority.
    
    Args:
        base_mean: Base mean processing time
    
    Returns:
        Function that takes a customer and returns processing time
    """
    def sample(customer):
        # Higher priority (lower value) gets faster service
        priority_factor = 1 + customer.priority
        return exponential(1 / (base_mean * priority_factor))
    
    return sample


# Time-dependent distributions
def time_varying_distribution(time_func: Callable[[float], float],
                             base_dist: Callable[[], float]) -> Callable[[float], float]:
    """
    Create a time-varying distribution.
    
    Args:
        time_func: Function that maps time to a scaling factor
        base_dist: Base distribution function
    
    Returns:
        Function that takes current time and returns a sample
    """
    def sample(current_time: float = 0):
        scale_factor = time_func(current_time)
        return base_dist() * scale_factor
    
    return sample


# Utility functions
def fit_exponential(data: List[float]) -> float:
    """Fit exponential distribution to data and return rate parameter."""
    return 1 / np.mean(data)


def fit_gamma(data: List[float]) -> tuple:
    """Fit gamma distribution to data and return (shape, scale) parameters."""
    mean = np.mean(data)
    var = np.var(data)
    
    shape = mean**2 / var
    scale = var / mean
    
    return shape, scale


def create_histogram_distribution(data: List[float], bins: int = 50) -> Callable[[], float]:
    """
    Create a distribution from histogram of data.
    
    Args:
        data: Sample data
        bins: Number of histogram bins
    
    Returns:
        Distribution function that samples from the histogram
    """
    hist, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Normalize histogram to create probabilities
    probs = hist / hist.sum()
    
    return lambda: np.random.choice(bin_centers, p=probs)
