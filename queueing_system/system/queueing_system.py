"""Main queueing system coordinator and simulation engine."""

import heapq
from typing import Callable, Dict, Optional, Set, Tuple
import numpy as np

from core import Component, Customer, SystemMetrics, Queue, Pool, MultiServerQueue


class QueueingSystem:
    """Main system wrapper that coordinates all components."""
    
    def __init__(self):
        self.components: Dict[str, Component] = {}  # {component_id: Component}
        self.routing_matrix: Dict[Tuple[str, str], float] = {}  # {(from_id, to_id): probability}
        self.sources: Dict[str, Tuple[Callable[[], float], Callable[[], Customer]]] = {}  # {source_id: (arrival_dist, customer_gen)}
        self.sinks: Set[str] = set()  # Component IDs that are exits
        self.current_time = 0.0
        self.customers: Dict[int, Customer] = {}  # {customer_id: Customer}
        self.next_customer_id = 0
        self.system_metrics = SystemMetrics()
        
        # Event heap: (time, event_type, component_id, customer_id)
        self.event_heap = []
        
    def add_component(self, component: Component) -> None:
        """Add a component to the system."""
        self.components[component.component_id] = component
    
    def add_source(self,
                   source_id: str,
                   arrival_distribution: Callable[[], float],
                   customer_generator: Optional[Callable[[], Customer]] = None) -> None:
        """Add an arrival source to the system."""
        if customer_generator is None:
            customer_generator = self._default_customer_generator
        self.sources[source_id] = (arrival_distribution, customer_generator)
    
    def add_sink(self, component_id: str) -> None:
        """Mark a component as a system exit."""
        self.sinks.add(component_id)
    
    def set_routing(self, from_id: str, to_id: str, probability: float = 1.0) -> None:
        """Set routing probability between components."""
        self.routing_matrix[(from_id, to_id)] = probability
    
    def _default_customer_generator(self) -> Customer:
        """Default customer generator."""
        customer = Customer(
            customer_id=self.next_customer_id,
            creation_time=self.current_time
        )
        self.next_customer_id += 1
        return customer
    
    def _schedule_next_arrival(self, source_id: str) -> None:
        """Schedule the next arrival from a source."""
        arrival_dist, _ = self.sources[source_id]
        next_arrival_time = self.current_time + arrival_dist()
        heapq.heappush(self.event_heap,
                      (next_arrival_time, 'arrival', source_id, None))
    
    def _route_customer(self, from_id: str) -> Optional[str]:
        """Determine where to route a customer from a component."""
        # Get all possible destinations and their probabilities
        destinations = [(to_id, prob)
                       for (f_id, to_id), prob in self.routing_matrix.items()
                       if f_id == from_id]
        
        if not destinations:
            return None
        
        # Probabilistic routing
        to_ids, probs = zip(*destinations)
        # Normalize probabilities if they don't sum to 1
        probs = np.array(probs)
        probs = probs / probs.sum()
        
        return np.random.choice(to_ids, p=probs)
    
    def simulate(self, max_time: float, warm_up_time: float = 0.0) -> None:
        """Run the simulation until max_time."""
        # Initialize sources
        for source_id in self.sources:
            self._schedule_next_arrival(source_id)
        
        # Main simulation loop
        while self.event_heap and self.event_heap[0][0] <= max_time:
            event_time, event_type, component_id, customer_id = heapq.heappop(self.event_heap)
            
            # Update system time
            self.current_time = event_time
            self.system_metrics.update_metrics(event_time)
            
            # Process warm-up period
            if event_time < warm_up_time:
                continue
                
            # Process event
            if event_type == 'arrival':
                self._process_arrival_event(component_id)
            elif event_type == 'departure':
                self._process_departure_event(component_id, customer_id)
        
        # Final metrics update
        self.system_metrics.update_metrics(max_time)
        self.current_time = max_time
    
    def _process_arrival_event(self, source_id: str) -> None:
        """Process an arrival from a source."""
        arrival_dist, customer_gen = self.sources[source_id]
        
        # Create new customer
        customer = customer_gen()
        self.customers[customer.customer_id] = customer
        self.system_metrics.total_customers_created += 1
        self.system_metrics.customers_in_system += 1
        
        # Route to first component
        first_component_id = self._route_customer(source_id)
        if first_component_id and first_component_id in self.components:
            component = self.components[first_component_id]
            accepted = component.process_arrival(customer, self.current_time)
            
            if accepted:
                # Check for immediate departure
                self._check_component_events(first_component_id)
            else:
                # Customer rejected due to capacity
                self.system_metrics.customers_in_system -= 1
        
        # Schedule next arrival
        self._schedule_next_arrival(source_id)
    
    def _process_departure_event(self, component_id: str, customer_id: int) -> None:
        """Process a departure from a component."""
        if component_id not in self.components:
            return
        
        component = self.components[component_id]
        
        # Process departure based on component type
        if isinstance(component, Queue):
            departing_customer = component.process_departure(self.current_time)
            # For queues, the departing customer is determined by the queue
            if departing_customer:
                customer_id = departing_customer.customer_id
        elif isinstance(component, Pool):
            departing_customer = component.process_departure(customer_id, self.current_time)
        elif isinstance(component, MultiServerQueue):
            departing_customer = component.process_departure(self.current_time)
            # For multi-server queues, the departing customer is determined by the queue
            if departing_customer:
                customer_id = departing_customer.customer_id
        else:
            # Generic component - implement if needed
            return
        
        if departing_customer is None:
            return
        
        # Check if this is a sink
        if component_id in self.sinks:
            self.system_metrics.total_customers_completed += 1
            self.system_metrics.customers_in_system -= 1
            self.system_metrics.total_system_time += (
                self.current_time - departing_customer.creation_time
            )
        else:
            # Route to next component
            next_component_id = self._route_customer(component_id)
            if next_component_id and next_component_id in self.components:
                next_component = self.components[next_component_id]
                accepted = next_component.process_arrival(departing_customer, self.current_time)
                
                if accepted:
                    self._check_component_events(next_component_id)
                else:
                    # Customer rejected at next component
                    self.system_metrics.customers_in_system -= 1
        
        # Check for more events from this component
        self._check_component_events(component_id)
    
    def _check_component_events(self, component_id: str) -> None:
        """Check if a component has any pending events."""
        component = self.components[component_id]
        next_event = component.get_next_event()
        
        if next_event is not None:
            event_time, event_type, customer = next_event
            heapq.heappush(self.event_heap,
                          (event_time, event_type, component_id, customer.customer_id))
    
    def get_metrics_summary(self) -> Dict:
        """Get a summary of all system metrics."""
        metrics = {
            'system': {
                'total_customers_created': self.system_metrics.total_customers_created,
                'total_customers_completed': self.system_metrics.total_customers_completed,
                'average_system_size': self.system_metrics.average_system_size(self.current_time),
                'average_system_time': self.system_metrics.average_system_time(),
                'throughput': self.system_metrics.throughput(self.current_time),
                'customers_in_system': self.system_metrics.customers_in_system
            },
            'components': {}
        }
        
        for comp_id, component in self.components.items():
            comp_metrics = {
                'total_arrivals': component.total_arrivals,
                'total_departures': component.total_departures,
                'average_size': component.average_size(),
                'current_size': component.current_size
            }
            
            if isinstance(component, Queue):
                comp_metrics['average_queue_time'] = component.average_queue_time()
                comp_metrics['utilization'] = component.utilization()
            elif isinstance(component, MultiServerQueue):
                comp_metrics['average_queue_time'] = component.average_queue_time()
                comp_metrics['server_utilization'] = component.server_utilization()
                comp_metrics['average_service_time'] = component.average_service_time()
            
            metrics['components'][comp_id] = comp_metrics
        
        return metrics
    
    def reset(self) -> None:
        """Reset the system to initial state."""
        self.current_time = 0.0
        self.customers.clear()
        self.next_customer_id = 0
        self.system_metrics = SystemMetrics()
        self.event_heap.clear()
        
        # Reset all components
        for component in self.components.values():
            component.current_time = 0.0
            component.total_arrivals = 0
            component.total_departures = 0
            component.current_size = 0
            component.size_time_product = 0.0
            component.last_update_time = 0.0
            
            if isinstance(component, Queue):
                component.queue.clear()
                component.in_service = None
                component.service_end_time = np.inf
                component.total_queue_time = 0.0
                component.customers_served = 0
            elif isinstance(component, Pool):
                component.customers.clear()
            elif isinstance(component, MultiServerQueue):
                component.queue.clear()
                component.servers = [None] * component.num_servers
                component.server_end_times = [np.inf] * component.num_servers
                component.total_queue_time = 0.0
                component.total_service_time = 0.0
                component.customers_served = 0
                component.server_busy_time = 0.0
