#!/usr/bin/env python3
"""
Run all combinations of distributions for Chick-fil-A model.
All times are in minutes.
"""

import sys
from pathlib import Path
import json
import datetime
import numpy as np
from itertools import product
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core import Queue, MultiServerQueue
from system import QueueingSystem
from distributions.random_variables import (
    gamma_distribution,
    lognormal_distribution,
    exponential_distribution,
    beta_distribution,
    weibull_distribution,
    pearson6_distribution
)

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# Define all distribution combinations
MOBILE_INTERARRIVAL_DISTS = {
    'lognormal': {
        'type': 'lognormal',
        'params': {'mean': -0.279, 'std': 1.46}
    },
    'pearson6': {
        'type': 'pearson6',
        'params': {'beta': 733, 'p': 0.75, 'q': 327, 'loc': 0}
    },
    'exponential': {
        'type': 'exponential',
        'params': {'rate': 1/1.68}  # beta = 1/rate
    }
}

DRIVE_THROUGH_INTERARRIVAL_DISTS = {
    'beta': {
        'type': 'beta',
        'params': {'a': 1.05, 'b': 4.01, 'loc': 0, 'scale': 3.65}
    },
    'weibull': {
        'type': 'weibull',
        'params': {'shape': 1.18, 'scale': 0.795}
    },
    'gamma': {
        'type': 'gamma',
        'params': {'shape': 1.27, 'scale': 0.543}
    },
    'exponential': {
        'type': 'exponential',
        'params': {'rate': 1/0.752}  # beta = 1/rate
    }
}

# Fixed distribution for drive-through service (order processing)
DRIVE_THROUGH_SERVICE_DIST = {
    'type': 'gamma',
    'params': {'shape': 6.16, 'scale': 0.17}
}

FOOD_SERVICE_DISTS = {
    'pearson6': {
        'type': 'pearson6',
        'params': {'beta': 4.79, 'p': 1.66, 'q': 16.7, 'loc': 0}
    },
    'gamma': {
        'type': 'gamma',
        'params': {'shape': 1.52, 'scale': 0.333}
    }
}


def create_distribution(dist_config):
    """Create a distribution function from configuration."""
    dist_type = dist_config['type']
    params = dist_config['params']
    
    if dist_type == 'lognormal':
        return lognormal_distribution(**params)
    elif dist_type == 'exponential':
        return exponential_distribution(**params)
    elif dist_type == 'gamma':
        return gamma_distribution(**params)
    elif dist_type == 'beta':
        # Beta needs special handling for loc and scale
        return lambda: np.random.beta(params['a'], params['b']) * params['scale'] + params.get('loc', 0)
    elif dist_type == 'weibull':
        return weibull_distribution(**params)
    elif dist_type == 'pearson6':
        return pearson6_distribution(**params)
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")


def create_chick_fil_a_model_from_config(config):
    """Create the Chick-fil-A model from a configuration."""
    system = QueueingSystem()
    
    # Create the main service queue (food preparation)
    food_service_dist = create_distribution(config['food_service'])
    service_queue = Queue(
        component_id="service",
        service_time_distribution=food_service_dist
    )
    system.add_component(service_queue)
    
    # Create the drive-through queue (2 servers for order taking)
    drive_through_service_dist = create_distribution(config['drive_through_service'])
    drive_through_queue = MultiServerQueue(
        component_id="drive_through",
        service_time_distribution=drive_through_service_dist,
        num_servers=2
    )
    system.add_component(drive_through_queue)
    
    # Add mobile orders source
    mobile_interarrival_dist = create_distribution(config['mobile_interarrival'])
    system.add_source(
        source_id="mobile_orders",
        arrival_distribution=mobile_interarrival_dist
    )
    
    # Add drive-through source
    drive_through_arrival_dist = create_distribution(config['drive_through_interarrival'])
    system.add_source(
        source_id="drive_through_arrivals",
        arrival_distribution=drive_through_arrival_dist
    )
    
    # Configure routing
    system.set_routing("mobile_orders", "service", 1.0)
    system.set_routing("drive_through_arrivals", "drive_through", 1.0)
    system.set_routing("drive_through", "service", 1.0)
    system.add_sink("service")
    
    return system


def run_simulation_with_config(config, simulation_time=120, random_seed=None):
    """Run a single simulation with the given configuration."""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    system = create_chick_fil_a_model_from_config(config)
    system.simulate(max_time=simulation_time)
    
    metrics = system.get_metrics_summary()
    
    # Add custom analysis
    drive_through_metrics = metrics['components']['drive_through']
    service_metrics = metrics['components']['service']
    
    # Calculate percentage of orders from each source
    drive_through_percentage = (drive_through_metrics['total_arrivals'] / 
                               metrics['system']['total_customers_created']) * 100
    
    metrics['custom'] = {
        'mobile_orders_percentage': 100 - drive_through_percentage,
        'drive_through_percentage': drive_through_percentage,
        'drive_through_server_utilization': drive_through_metrics.get('server_utilization', 0),
        'total_system_throughput': service_metrics['total_departures'] / simulation_time,
        'simulation_time_minutes': simulation_time
    }
    
    return metrics


def generate_all_combinations():
    """Generate all combinations of distributions."""
    combinations = []
    
    for mobile_name, mobile_dist in MOBILE_INTERARRIVAL_DISTS.items():
        for dt_name, dt_dist in DRIVE_THROUGH_INTERARRIVAL_DISTS.items():
            for food_name, food_dist in FOOD_SERVICE_DISTS.items():
                config = {
                    'mobile_interarrival': mobile_dist,
                    'drive_through_interarrival': dt_dist,
                    'drive_through_service': DRIVE_THROUGH_SERVICE_DIST,
                    'food_service': food_dist,
                    'name': f"mobile_{mobile_name}_dt_{dt_name}_food_{food_name}"
                }
                combinations.append(config)
    
    return combinations


def main():
    """Run all combinations and save results."""
    # Create results directory
    results_dir = Path("results") / f"chick_fil_a_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all combinations
    combinations = generate_all_combinations()
    print(f"Running {len(combinations)} distribution combinations...")
    
    # Run simulations
    all_results = {}
    summary_results = []
    
    for i, config in enumerate(combinations):
        print(f"\nRunning combination {i+1}/{len(combinations)}: {config['name']}")
        
        # Run multiple replications for statistical significance
        replications = 10
        replication_results = []
        
        for rep in range(replications):
            seed = 42 + i * 1000 + rep  # Unique seed for each combination and replication
            metrics = run_simulation_with_config(config, simulation_time=120, random_seed=seed)
            replication_results.append(metrics)
        
        # Calculate summary statistics
        summary = calculate_summary_statistics(replication_results)
        summary['config'] = config
        summary['name'] = config['name']
        
        # Save individual results
        individual_file = results_dir / f"{config['name']}.json"
        results_data = {
            'config': config,
            'replications': replication_results,
            'summary': summary
        }
        # Convert numpy types before saving
        results_data = convert_numpy_types(results_data)
        with open(individual_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        all_results[config['name']] = {
            'config': config,
            'summary': summary
        }
        
        # Add to summary list
        summary_results.append({
            'name': config['name'],
            'throughput_mean': summary['system']['throughput']['mean'],
            'throughput_std': summary['system']['throughput']['std'],
            'mobile_percentage': summary['custom']['mobile_orders_percentage']['mean'],
            'dt_utilization': summary['custom']['drive_through_server_utilization']['mean'],
            'service_queue_size': summary['components']['service']['average_size']['mean'],
            'dt_queue_size': summary['components']['drive_through']['average_size']['mean'],
            'service_wait_time': summary['components']['service']['average_queue_time']['mean']
        })
        
        # Print summary
        print(f"  Throughput: {summary['system']['throughput']['mean']:.2f} ± "
              f"{summary['system']['throughput']['std']:.2f} customers/minute")
        print(f"  Mobile %: {summary['custom']['mobile_orders_percentage']['mean']:.1f}%")
        print(f"  DT Utilization: {summary['custom']['drive_through_server_utilization']['mean']:.1%}")
    
    # Save overall summary
    summary_file = results_dir / "summary.json"
    # Convert numpy types before saving
    all_results_converted = convert_numpy_types(all_results)
    with open(summary_file, 'w') as f:
        json.dump(all_results_converted, f, indent=2)
    
    # Create comparison CSV
    import pandas as pd
    comparison_df = pd.DataFrame(summary_results)
    comparison_df.to_csv(results_dir / "comparison.csv", index=False)
    
    # Print final summary
    print("\n=== Final Summary ===")
    print(f"Results saved to: {results_dir}")
    print("\nTop 5 configurations by throughput:")
    comparison_df_sorted = comparison_df.sort_values('throughput_mean', ascending=False)
    print(comparison_df_sorted[['name', 'throughput_mean', 'mobile_percentage', 
                               'dt_utilization', 'service_wait_time']].head())
    
    # Create visualization
    create_comparison_plots(comparison_df, results_dir)


def calculate_summary_statistics(replication_results):
    """Calculate summary statistics across replications."""
    summary = {
        'system': {},
        'components': {},
        'custom': {}
    }
    
    # System metrics
    system_keys = replication_results[0]['system'].keys()
    for key in system_keys:
        values = [r['system'][key] for r in replication_results]
        summary['system'][key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    # Component metrics
    component_ids = replication_results[0]['components'].keys()
    for comp_id in component_ids:
        summary['components'][comp_id] = {}
        metric_keys = replication_results[0]['components'][comp_id].keys()
        
        for key in metric_keys:
            values = [r['components'][comp_id][key] for r in replication_results]
            summary['components'][comp_id][key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    # Custom metrics
    custom_keys = replication_results[0]['custom'].keys()
    for key in custom_keys:
        values = [r['custom'][key] for r in replication_results]
        summary['custom'][key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    return summary


def create_comparison_plots(df, results_dir):
    """Create comparison plots for the results."""
    import matplotlib.pyplot as plt
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Sort by throughput
    df_sorted = df.sort_values('throughput_mean', ascending=False)
    
    # Plot 1: Throughput comparison
    ax1.bar(range(len(df_sorted)), df_sorted['throughput_mean'], 
            yerr=df_sorted['throughput_std'], capsize=3)
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Throughput (customers/minute)')
    ax1.set_title('Throughput Comparison')
    ax1.tick_params(axis='x', rotation=90)
    ax1.set_xticks(range(len(df_sorted)))
    ax1.set_xticklabels(df_sorted['name'], rotation=90, ha='right')
    
    # Plot 2: Mobile percentage vs throughput
    scatter = ax2.scatter(df['mobile_percentage'], df['throughput_mean'], 
                         c=df['dt_utilization'], cmap='viridis')
    ax2.set_xlabel('Mobile Order Percentage (%)')
    ax2.set_ylabel('Throughput (customers/minute)')
    ax2.set_title('Mobile % vs Throughput (color = DT Utilization)')
    plt.colorbar(scatter, ax=ax2, label='DT Utilization')
    
    # Plot 3: Queue sizes
    x = np.arange(len(df_sorted))
    width = 0.35
    ax3.bar(x - width/2, df_sorted['service_queue_size'], width, label='Service Queue')
    ax3.bar(x + width/2, df_sorted['dt_queue_size'], width, label='Drive-Through Queue')
    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('Average Queue Size')
    ax3.set_title('Queue Sizes by Configuration')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=90)
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_sorted['name'], rotation=90, ha='right')
    
    # Plot 4: Wait times vs utilization
    ax4.scatter(df['dt_utilization'] * 100, df['service_wait_time'])
    ax4.set_xlabel('Drive-Through Utilization (%)')
    ax4.set_ylabel('Service Wait Time (minutes)')
    ax4.set_title('Service Wait Time vs DT Utilization')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create heatmap of key metrics
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for heatmap
    metrics_matrix = df[['throughput_mean', 'mobile_percentage', 'dt_utilization', 
                        'service_wait_time', 'service_queue_size']].values.T
    
    im = ax.imshow(metrics_matrix, aspect='auto', cmap='coolwarm')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(df)))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels(df['name'], rotation=90, ha='right')
    ax.set_yticklabels(['Throughput', 'Mobile %', 'DT Util', 'Service Wait', 'Service Queue'])
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Add text annotations
    for i in range(5):
        for j in range(len(df)):
            text = ax.text(j, i, f'{metrics_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Metrics Heatmap')
    plt.tight_layout()
    plt.savefig(results_dir / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
