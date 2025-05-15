#!/usr/bin/env python3
"""Command-line interface for running queueing simulations."""

import argparse
import json
import sys
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

# Since we're now using main.py, imports are simpler
from core import Queue, Pool
from system import QueueingSystem
from distributions.random_variables import (
    exponential_distribution,
    uniform_distribution,
    normal_distribution,
    gamma_distribution,
    deterministic_distribution
)
from visualization.plotting import (
    plot_system_metrics,
    create_performance_report
)

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from core import Queue, Pool
from system import QueueingSystem
from distributions.random_variables import (
    exponential_distribution,
    uniform_distribution,
    normal_distribution,
    gamma_distribution,
    deterministic_distribution
)
from visualization.plotting import (
    plot_system_metrics,
    create_performance_report
)


def create_mm1_system(arrival_rate: float, service_rate: float) -> QueueingSystem:
    """Create a simple M/M/1 queue system."""
    system = QueueingSystem()
    
    # Create queue
    queue = Queue(
        component_id="main_queue",
        service_time_distribution=exponential_distribution(service_rate)
    )
    system.add_component(queue)
    
    # Add source
    system.add_source(
        source_id="arrivals",
        arrival_distribution=exponential_distribution(arrival_rate)
    )
    
    # Configure routing
    system.set_routing("arrivals", "main_queue", 1.0)
    system.add_sink("main_queue")
    
    return system


def create_tandem_system(arrival_rate: float, 
                        service_rates: List[float]) -> QueueingSystem:
    """Create a tandem queue system."""
    system = QueueingSystem()
    
    # Create queues
    for i, rate in enumerate(service_rates):
        queue = Queue(
            component_id=f"queue_{i+1}",
            service_time_distribution=exponential_distribution(rate)
        )
        system.add_component(queue)
    
    # Add source
    system.add_source(
        source_id="arrivals",
        arrival_distribution=exponential_distribution(arrival_rate)
    )
    
    # Configure routing
    system.set_routing("arrivals", "queue_1", 1.0)
    for i in range(len(service_rates) - 1):
        system.set_routing(f"queue_{i+1}", f"queue_{i+2}", 1.0)
    system.add_sink(f"queue_{len(service_rates)}")
    
    return system


def create_network_system(config: Dict[str, Any]) -> QueueingSystem:
    """Create a network system from configuration."""
    system = QueueingSystem()
    
    # Create components
    for comp_config in config['components']:
        comp_type = comp_config['type']
        comp_id = comp_config['id']
        
        if comp_type == 'queue':
            dist_type = comp_config.get('distribution', 'exponential')
            rate = comp_config.get('rate', 1.0)
            capacity = comp_config.get('capacity', np.inf)
            
            if dist_type == 'exponential':
                dist = exponential_distribution(rate)
            elif dist_type == 'uniform':
                a = comp_config.get('min', 0)
                b = comp_config.get('max', 2/rate)
                dist = uniform_distribution(a, b)
            elif dist_type == 'deterministic':
                dist = deterministic_distribution(1/rate)
            else:
                dist = exponential_distribution(rate)
            
            queue = Queue(component_id=comp_id, 
                         service_time_distribution=dist,
                         capacity=capacity)
            system.add_component(queue)
            
        elif comp_type == 'pool':
            dist_type = comp_config.get('distribution', 'exponential')
            rate = comp_config.get('rate', 1.0)
            capacity = comp_config.get('capacity', np.inf)
            selection = comp_config.get('selection', 'fcfs')
            
            if dist_type == 'exponential':
                dist = lambda customer: exponential_distribution(rate)()
            else:
                dist = lambda customer: exponential_distribution(rate)()
            
            pool = Pool(component_id=comp_id,
                       exit_time_distribution=dist,
                       capacity=capacity,
                       selection_method=selection)
            system.add_component(pool)
    
    # Add sources
    for source_config in config['sources']:
        source_id = source_config['id']
        rate = source_config.get('rate', 1.0)
        system.add_source(source_id, exponential_distribution(rate))
    
    # Configure routing
    for route in config['routing']:
        from_id = route['from']
        to_id = route['to']
        prob = route.get('probability', 1.0)
        system.set_routing(from_id, to_id, prob)
    
    # Add sinks
    for sink_id in config.get('sinks', []):
        system.add_sink(sink_id)
    
    return system


def run_simulation(system: QueueingSystem, 
                  simulation_time: float,
                  warm_up_time: float = 0.0,
                  random_seed: Optional[int] = None) -> Dict:
    """Run a single simulation and return metrics."""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    system.reset()
    system.simulate(max_time=simulation_time, warm_up_time=warm_up_time)
    return system.get_metrics_summary()


def run_replications(system: QueueingSystem,
                    simulation_time: float,
                    num_replications: int,
                    warm_up_time: float = 0.0,
                    base_seed: int = 42) -> Dict:
    """Run multiple replications and compute statistics."""
    results = []
    
    for i in range(num_replications):
        seed = base_seed + i
        metrics = run_simulation(system, simulation_time, warm_up_time, seed)
        results.append(metrics)
    
    # Compute summary statistics
    summary = {
        'replications': num_replications,
        'simulation_time': simulation_time,
        'warm_up_time': warm_up_time,
        'system': {},
        'components': {}
    }
    
    # System-level statistics
    system_keys = results[0]['system'].keys()
    for key in system_keys:
        values = [r['system'][key] for r in results]
        summary['system'][key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    # Component-level statistics
    component_ids = results[0]['components'].keys()
    for comp_id in component_ids:
        summary['components'][comp_id] = {}
        metric_keys = results[0]['components'][comp_id].keys()
        
        for key in metric_keys:
            values = [r['components'][comp_id][key] for r in results]
            summary['components'][comp_id][key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    return summary


def save_results(results: Dict, output_path: str) -> None:
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def print_results(results: Dict, detailed: bool = False) -> None:
    """Print results to console."""
    print("\n=== Simulation Results ===")
    
    if 'replications' in results:
        # Summary statistics from multiple runs
        print(f"Replications: {results['replications']}")
        print(f"Simulation Time: {results['simulation_time']}")
        print(f"Warm-up Time: {results['warm_up_time']}")
        
        print("\nSystem Metrics:")
        for metric, stats in results['system'].items():
            print(f"  {metric}:")
            print(f"    Mean: {stats['mean']:.4f} (±{stats['std']:.4f})")
            if detailed:
                print(f"    Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
        
        print("\nComponent Metrics:")
        for comp_id, metrics in results['components'].items():
            print(f"  {comp_id}:")
            for metric, stats in metrics.items():
                print(f"    {metric}: {stats['mean']:.4f} (±{stats['std']:.4f})")
    else:
        # Single run results
        print("\nSystem Metrics:")
        for metric, value in results['system'].items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nComponent Metrics:")
        for comp_id, metrics in results['components'].items():
            print(f"  {comp_id}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description='Run queueing simulations')
    
    # Simulation type
    parser.add_argument('type', choices=['mm1', 'tandem', 'network', 'custom'],
                       help='Type of queueing system to simulate')
    
    # Common parameters
    parser.add_argument('-t', '--time', type=float, default=10000,
                       help='Simulation time (default: 10000)')
    parser.add_argument('-w', '--warmup', type=float, default=0,
                       help='Warm-up time (default: 0)')
    parser.add_argument('-r', '--replications', type=int, default=1,
                       help='Number of replications (default: 1)')
    parser.add_argument('-s', '--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # System-specific parameters
    parser.add_argument('--arrival-rate', type=float, default=0.8,
                       help='Arrival rate for simple systems (default: 0.8)')
    parser.add_argument('--service-rate', type=float, default=1.0,
                       help='Service rate for M/M/1 (default: 1.0)')
    parser.add_argument('--service-rates', type=float, nargs='+',
                       help='Service rates for tandem queues')
    parser.add_argument('--config', type=str,
                       help='Configuration file for network systems')
    
    # Output options
    parser.add_argument('-o', '--output', type=str,
                       help='Output file for results (JSON)')
    parser.add_argument('-p', '--plot', action='store_true',
                       help='Generate plots')
    parser.add_argument('--plot-file', type=str,
                       help='Save plots to file')
    parser.add_argument('-d', '--detailed', action='store_true',
                       help='Show detailed statistics')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress console output')
    
    args = parser.parse_args()
    
    # Create system based on type
    if args.type == 'mm1':
        system = create_mm1_system(args.arrival_rate, args.service_rate)
    elif args.type == 'tandem':
        if not args.service_rates:
            args.service_rates = [1.2, 1.0]  # Default tandem rates
        system = create_tandem_system(args.arrival_rate, args.service_rates)
    elif args.type == 'network':
        if not args.config:
            print("Error: Network system requires --config file")
            sys.exit(1)
        with open(args.config, 'r') as f:
            config = json.load(f)
        system = create_network_system(config)
    else:
        print("Error: Custom systems not yet implemented")
        sys.exit(1)
    
    # Run simulation(s)
    if args.replications > 1:
        results = run_replications(system, args.time, args.replications, 
                                 args.warmup, args.seed)
    else:
        results = run_simulation(system, args.time, args.warmup, args.seed)
    
    # Output results
    if not args.quiet:
        print_results(results, args.detailed)
    
    if args.output:
        save_results(results, args.output)
        if not args.quiet:
            print(f"\nResults saved to: {args.output}")
    
    # Generate plots
    if args.plot or args.plot_file:
        # For plotting, we need a single run
        if args.replications > 1:
            print("\nRunning additional simulation for plotting...")
            plot_results = run_simulation(system, args.time, args.warmup, args.seed)
        else:
            plot_results = results
        
        fig = plot_system_metrics(plot_results)
        
        if args.plot_file:
            fig.savefig(args.plot_file, dpi=300, bbox_inches='tight')
            if not args.quiet:
                print(f"Plot saved to: {args.plot_file}")
        
        if args.plot:
            import matplotlib.pyplot as plt
            plt.show()


if __name__ == '__main__':
    main()
