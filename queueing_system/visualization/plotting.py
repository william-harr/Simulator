"""
Visualization utilities for queueing systems.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import seaborn as sns


def plot_system_metrics(metrics: dict, title: str = "Queueing System Metrics"):
    """Create a comprehensive dashboard of system metrics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # System-level metrics
    system_metrics = metrics['system']
    
    # Plot 1: Throughput over time (if available)
    ax1.bar(['Created', 'Completed'], 
            [system_metrics['total_customers_created'], 
             system_metrics['total_customers_completed']])
    ax1.set_ylabel('Number of Customers')
    ax1.set_title('Customer Flow')
    
    # Plot 2: Average metrics
    ax2.bar(['System Size', 'System Time'], 
            [system_metrics['average_system_size'], 
             system_metrics['average_system_time']])
    ax2.set_ylabel('Average Value')
    ax2.set_title('System Averages')
    
    # Plot 3: Component sizes
    component_ids = list(metrics['components'].keys())
    avg_sizes = [metrics['components'][comp]['average_size'] 
                 for comp in component_ids]
    
    ax3.bar(component_ids, avg_sizes)
    ax3.set_ylabel('Average Size')
    ax3.set_xlabel('Component')
    ax3.set_title('Component Average Sizes')
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Component-specific metrics
    queue_times = []
    queue_labels = []
    
    for comp_id, comp_metrics in metrics['components'].items():
        if 'average_queue_time' in comp_metrics:
            queue_times.append(comp_metrics['average_queue_time'])
            queue_labels.append(comp_id)
    
    if queue_times:
        ax4.bar(queue_labels, queue_times)
        ax4.set_ylabel('Average Queue Time')
        ax4.set_xlabel('Queue')
        ax4.set_title('Queue Waiting Times')
        ax4.tick_params(axis='x', rotation=45)
    else:
        ax4.text(0.5, 0.5, 'No Queue Time Data', 
                 ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    return fig


def plot_customer_journey(system, customer_id: int):
    """Plot the journey of a specific customer through the system."""
    if customer_id not in system.customers:
        print(f"Customer {customer_id} not found")
        return None
    
    customer = system.customers[customer_id]
    
    # Extract journey data
    components = []
    arrival_times = []
    departure_times = []
    
    for comp_id in customer.arrival_times:
        components.append(comp_id)
        arrival_times.append(customer.arrival_times[comp_id])
        if comp_id in customer.departure_times:
            departure_times.append(customer.departure_times[comp_id])
        else:
            departure_times.append(system.current_time)  # Still in system
    
    # Create Gantt-like chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, comp in enumerate(components):
        ax.barh(i, departure_times[i] - arrival_times[i], 
                left=arrival_times[i], height=0.5,
                label=comp if i == 0 else "")
    
    ax.set_yticks(range(len(components)))
    ax.set_yticklabels(components)
    ax.set_xlabel('Time')
    ax.set_title(f'Customer {customer_id} Journey Through System')
    ax.legend()
    
    return fig


def plot_utilization_over_time(system, component_id: str, 
                              num_intervals: int = 100):
    """Plot component utilization over time (requires event history)."""
    # This is a placeholder - would need to track utilization history
    # during simulation for accurate plotting
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate sample data for demonstration
    times = np.linspace(0, system.current_time, num_intervals)
    utilization = np.random.beta(2, 5, size=num_intervals)  # Sample data
    
    ax.plot(times, utilization)
    ax.set_xlabel('Time')
    ax.set_ylabel('Utilization')
    ax.set_title(f'Utilization of {component_id} Over Time')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_distribution_comparison(simulated_data: List[float], 
                                theoretical_func: callable,
                                title: str = "Distribution Comparison"):
    """Compare simulated data with theoretical distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of simulated data
    ax1.hist(simulated_data, bins=50, density=True, alpha=0.7, 
             color='blue', label='Simulated')
    
    # Theoretical distribution
    x_range = np.linspace(min(simulated_data), max(simulated_data), 100)
    theoretical_values = [theoretical_func(x) for x in x_range]
    ax1.plot(x_range, theoretical_values, 'r-', linewidth=2, 
             label='Theoretical')
    
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title('PDF Comparison')
    ax1.legend()
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(simulated_data, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot')
    
    fig.suptitle(title)
    return fig


def plot_network_topology(system):
    """Visualize the network topology of the queueing system."""
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX required for topology visualization")
        return None
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for source_id in system.sources:
        G.add_node(source_id, node_type='source')
    
    for comp_id in system.components:
        if comp_id in system.sinks:
            G.add_node(comp_id, node_type='sink')
        else:
            G.add_node(comp_id, node_type='component')
    
    # Add edges from routing matrix
    for (from_id, to_id), prob in system.routing_matrix.items():
        G.add_edge(from_id, to_id, weight=prob)
    
    # Create layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw nodes by type
    sources = [n for n in G.nodes if G.nodes[n].get('node_type') == 'source']
    components = [n for n in G.nodes if G.nodes[n].get('node_type') == 'component']
    sinks = [n for n in G.nodes if G.nodes[n].get('node_type') == 'sink']
    
    nx.draw_networkx_nodes(G, pos, sources, node_color='green', 
                          node_size=500, label='Sources', ax=ax)
    nx.draw_networkx_nodes(G, pos, components, node_color='blue', 
                          node_size=500, label='Components', ax=ax)
    nx.draw_networkx_nodes(G, pos, sinks, node_color='red', 
                          node_size=500, label='Sinks', ax=ax)
    
    # Draw edges with probabilities
    edge_labels = {(u, v): f"{d['weight']:.2f}" 
                   for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, 
                          arrowsize=20, edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax)
    
    ax.set_title('Queueing Network Topology')
    ax.legend()
    ax.axis('off')
    
    return fig


def create_performance_report(system, save_path: Optional[str] = None):
    """Create a comprehensive performance report with multiple visualizations."""
    metrics = system.get_metrics_summary()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 20))
    
    # Main metrics dashboard
    plt.subplot(3, 1, 1)
    plot_system_metrics(metrics)
    
    # Network topology (if networkx available)
    plt.subplot(3, 2, 3)
    try:
        plot_network_topology(system)
    except:
        plt.text(0.5, 0.5, 'Network topology visualization not available', 
                ha='center', va='center')
    
    # Component performance comparison
    plt.subplot(3, 2, 4)
    comp_names = list(metrics['components'].keys())
    arrivals = [metrics['components'][c]['total_arrivals'] for c in comp_names]
    departures = [metrics['components'][c]['total_departures'] for c in comp_names]
    
    x = np.arange(len(comp_names))
    width = 0.35
    
    plt.bar(x - width/2, arrivals, width, label='Arrivals')
    plt.bar(x + width/2, departures, width, label='Departures')
    plt.xlabel('Component')
    plt.ylabel('Count')
    plt.title('Component Throughput')
    plt.xticks(x, comp_names, rotation=45)
    plt.legend()
    
    # System statistics summary
    plt.subplot(3, 1, 3)
    plt.axis('off')
    
    stats_text = f"""
    System Performance Summary
    -------------------------
    Total Simulation Time: {system.current_time:.2f}
    Customers Created: {metrics['system']['total_customers_created']}
    Customers Completed: {metrics['system']['total_customers_completed']}
    System Throughput: {metrics['system']['throughput']:.4f} customers/time
    Average System Size: {metrics['system']['average_system_size']:.4f}
    Average System Time: {metrics['system']['average_system_time']:.4f}
    
    Component Details:
    """
    
    for comp_id, comp_metrics in metrics['components'].items():
        stats_text += f"\n{comp_id}:"
        stats_text += f"\n  Average Size: {comp_metrics['average_size']:.4f}"
        stats_text += f"\n  Current Size: {comp_metrics['current_size']}"
        if 'average_queue_time' in comp_metrics:
            stats_text += f"\n  Average Queue Time: {comp_metrics['average_queue_time']:.4f}"
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
             fontfamily='monospace', verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig