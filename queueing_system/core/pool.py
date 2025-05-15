"""Pool component implementation."""

from typing import Callable, Dict, Optional, Tuple
import numpy as np

from core.base import Component, Customer


class Pool(Component):
    """Pool where any customer can be selected for departure."""
    
    def __init__(self,
                 component_id: str,
                 exit_time_distribution: Callable[[Customer], float],
                 capacity: float = np.inf,
                 selection_method: str = 'fcfs'):
        super().__init__(component_id)
        self.exit_time_distribution = exit_time_distribution
        self.capacity = capacity
        self.selection_method = selection_method
        self.customers: Dict[int, Tuple[Customer, float]] = {}  # {customer_id: (customer, exit_time)}
        
    def process_arrival(self, customer: Customer, current_time: float) -> bool:
        """Process an arriving customer."""
        self.update_size_metrics(current_time)
        
        # Check capacity
        if self.current_size >= self.capacity:
            return False
        
        # Record arrival
        customer.arrival_times[self.component_id] = current_time
        self.total_arrivals += 1
        
        # Calculate exit time for this customer
        exit_time = current_time + self.exit_time_distribution(customer)
        self.customers[customer.customer_id] = (customer, exit_time)
        
        self.current_size += 1
        return True
    
    def get_next_event(self) -> Optional[Tuple[float, str, Customer]]:
        """Get the next departure event based on selection method."""
        if not self.customers:
            return None
        
        if self.selection_method == 'fcfs':
            # First come, first served - earliest exit time
            min_customer = min(self.customers.values(), key=lambda x: x[1])
            return (min_customer[1], 'departure', min_customer[0])
        
        elif self.selection_method == 'priority':
            # Priority-based selection - lower priority value = higher priority
            min_customer = min(self.customers.values(), 
                             key=lambda x: (x[0].priority, x[1]))
            return (min_customer[1], 'departure', min_customer[0])
        
        elif self.selection_method == 'random':
            # Random selection - need to determine next selection time
            if self.customers:
                # Simple approach: use minimum exit time among all customers
                min_customer = min(self.customers.values(), key=lambda x: x[1])
                return (min_customer[1], 'departure', min_customer[0])
        
        return None
    
    def process_departure(self, customer_id: int, current_time: float) -> Optional[Customer]:
        """Process a departure from the pool."""
        self.update_size_metrics(current_time)
        
        if customer_id not in self.customers:
            return None
        
        customer, _ = self.customers.pop(customer_id)
        customer.departure_times[self.component_id] = current_time
        self.total_departures += 1
        self.current_size -= 1
        
        return customer
    
    def average_wait_time(self) -> float:
        """Calculate average waiting time in pool."""
        if self.total_departures > 0:
            total_wait = sum(
                c.departure_times.get(self.component_id, 0) - 
                c.arrival_times.get(self.component_id, 0)
                for c, _ in self.customers.values()
                if self.component_id in c.departure_times
            )
            return total_wait / self.total_departures
        return 0.0
