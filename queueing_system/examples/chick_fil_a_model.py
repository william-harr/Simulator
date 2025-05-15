"""Chick-fil-A queueing model with drive-through and mobile orders."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from core import Queue, MultiServerQueue
from system import QueueingSystem
from distributions.random_variables import (
    gamma_distribution,
    lognormal_distribution,
    exponential_distribution,
    beta_distribution,
    pearson3_distribution
)


def create_chick_fil_a_model(
    mobile_mean: float = 2.0,  # Log-normal mean (not actual mean)
    mobile_std: float = 0.5,   # Log-normal std
    drive_through_rate: float = 2.0,  # 2 customers per time unit
    drive_through_dist: str = 'exponential',
    drive_through_params: dict = None,
    service_dist: str = 'gamma',
    service_params: dict = None,
    drive_through_service_params: dict = None
) -> QueueingSystem:
    """
    Create the Chick-fil-A queueing model.
    
    Note: For log-normal distribution, the actual mean interarrival time is:
    exp(mobile_mean + mobile_std^2/2)
    """
    system = QueueingSystem()
    
    # Default parameters if not provided
    if drive_through_params is None:
        drive_through_params = {'rate': drive_through_rate}
    
    if service_params is None:
        # Service time around 0.5 time units on average
        service_params = {'shape': 2.0, 'scale': 0.25}
    
    if drive_through_service_params is None:
        # Drive-through service time around 0.4 time units on average
        drive_through_service_params = {'shape': 2.0, 'scale': 0.2}
    
    # Create the main service queue
    if service_dist == 'gamma':
        service_time_dist = gamma_distribution(
            service_params['shape'],
            service_params['scale']
        )
    elif service_dist == 'pearson':
        service_time_dist = pearson3_distribution(
            service_params.get('shape', 2.0),
            service_params.get('scale', 0.5),
            service_params.get('loc', 0)
        )
    else:
        raise ValueError(f"Unknown service distribution: {service_dist}")
    
    service_queue = Queue(
        component_id="service",
        service_time_distribution=service_time_dist
    )
    system.add_component(service_queue)
    
    # Create the drive-through queue (2 servers)
    drive_through_service_dist = gamma_distribution(
        drive_through_service_params['shape'],
        drive_through_service_params['scale']
    )
    
    drive_through_queue = MultiServerQueue(
        component_id="drive_through",
        service_time_distribution=drive_through_service_dist,
        num_servers=2
    )
    system.add_component(drive_through_queue)
    
    # Add mobile orders source (log-normal interarrival)
    mobile_interarrival_dist = lognormal_distribution(mobile_mean, mobile_std)
    system.add_source(
        source_id="mobile_orders",
        arrival_distribution=mobile_interarrival_dist
    )
    
    # Add drive-through source
    if drive_through_dist == 'exponential':
        dt_arrival_dist = exponential_distribution(drive_through_params['rate'])
    elif drive_through_dist == 'gamma':
        dt_arrival_dist = gamma_distribution(
            drive_through_params.get('shape', 2.0),
            drive_through_params.get('scale', 1/drive_through_rate)
        )
    elif drive_through_dist == 'beta':
        dt_arrival_dist = beta_distribution(
            drive_through_params.get('a', 2.0),
            drive_through_params.get('b', 5.0)
        )
    elif drive_through_dist == 'pearson':
        dt_arrival_dist = pearson3_distribution(
            drive_through_params.get('shape', 2.0),
            drive_through_params.get('scale', 1/drive_through_rate),
            drive_through_params.get('loc', 0)
        )
    else:
        raise ValueError(f"Unknown drive-through distribution: {drive_through_dist}")
    
    system.add_source(
        source_id="drive_through_arrivals",
        arrival_distribution=dt_arrival_dist
    )
    
    # Configure routing
    # Mobile orders go directly to service
    system.set_routing("mobile_orders", "service", 1.0)
    
    # Drive-through arrivals go to drive-through queue
    system.set_routing("drive_through_arrivals", "drive_through", 1.0)
    
    # Drive-through completions go to service
    system.set_routing("drive_through", "service", 1.0)
    
    # Service is the final sink
    system.add_sink("service")
    
    return system


def run_chick_fil_a_simulation(
    simulation_time: float = 3600,  # 1 hour in seconds
    **model_params
) -> dict:
    """Run the Chick-fil-A simulation and return results."""
    
    # Create the model
    system = create_chick_fil_a_model(**model_params)
    
    # Run simulation
    system.simulate(max_time=simulation_time)
    
    # Get metrics
    metrics = system.get_metrics_summary()
    
    # Add some custom analysis
    drive_through_metrics = metrics['components']['drive_through']
    service_metrics = metrics['components']['service']
    
    # Calculate percentage of orders from each source
    mobile_orders_count = metrics['system']['total_customers_created'] - drive_through_metrics['total_arrivals']
    drive_through_percentage = (drive_through_metrics['total_arrivals'] /
                               metrics['system']['total_customers_created']) * 100
    
    metrics['custom'] = {
        'mobile_orders_percentage': 100 - drive_through_percentage,
        'drive_through_percentage': drive_through_percentage,
        'drive_through_server_utilization': drive_through_metrics.get('server_utilization', 0),
        'total_system_throughput': service_metrics['total_departures'] / simulation_time * 60  # per minute
    }
    
    return metrics


if __name__ == "__main__":
    results = run_chick_fil_a_simulation(
        simulation_time= 120,        # Minutes
        mobile_mean= -.279,      # Log-normal parameters
        mobile_std=1.46,       # Results in mean interarrival ≈ 8.8
        drive_through_rate=2.0,  # 2 customers per time unit
        drive_through_dist='exponential',
        service_dist='gamma',
        service_params={'shape': 2.0, 'scale': 0.25},  # Mean service time = 0.5
        drive_through_service_params={'shape': 2.0, 'scale': 0.2}  # Mean = 0.4
    )
    
    # Add debug information
    print("\n=== Debug Information ===")
    print(f"Total arrivals to drive-through: {results['components']['drive_through']['total_arrivals']}")
    print(f"Total departures from drive-through: {results['components']['drive_through']['total_departures']}")
    print(f"Current drive-through size: {results['components']['drive_through']['current_size']}")
    
    print("\n=== Chick-fil-A Model Results ===")
    print(f"Total customers served: {results['system']['total_customers_completed']}")
    print(f"Mobile orders: {results['custom']['mobile_orders_percentage']:.1f}%")
    print(f"Drive-through orders: {results['custom']['drive_through_percentage']:.1f}%")
    print(f"System throughput: {results['custom']['total_system_throughput']:.2f} customers/minute")
    print("\nDrive-through queue:")
    print(f"  Average size: {results['components']['drive_through']['average_size']:.2f}")
    print(f"  Server utilization: {results['custom']['drive_through_server_utilization']:.2%}")
    print("\nService queue:")
    print(f"  Average size: {results['components']['service']['average_size']:.2f}")
    print(f"  Average wait time: {results['components']['service']['average_queue_time']:.2f}")
