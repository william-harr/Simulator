"""Queue component implementation."""

from collections import deque
from typing import Callable, Optional, Tuple
import numpy as np

from core.base import Component, Customer


class Queue(Component):
    """General FIFO queue with service at the front."""
    
    def __init__(self, 
                 component_id: str,
                 service_time_distribution: Callable[[], float],
                 capacity: float = np.inf):
        super().__init__(component_id)
        self.service_time_distribution = service_time_distribution
        self.capacity = capacity
        self.queue = deque()
        self.in_service = None
        self.service_end_time = np.inf
        
        # Additional metrics
        self.total_queue_time = 0.0
        self.customers_served = 0
        
    def process_arrival(self, customer: Customer, current_time: float) -> bool:
        """Process an arriving customer."""
        self.update_size_metrics(current_time)
        
        # Check capacity
        if self.current_size >= self.capacity:
            return False  # Reject customer
        
        # Record arrival
        customer.arrival_times[self.component_id] = current_time
        customer.queue_entry_times[self.component_id] = current_time
        self.total_arrivals += 1
        
        # If no one is being served, start service immediately
        if self.in_service is None:
            self.in_service = customer
            service_time = self.service_time_distribution()
            self.service_end_time = current_time + service_time
        else:
            # Add to queue
            self.queue.append(customer)
        
        self.current_size += 1
        return True
    
    def get_next_event(self) -> Optional[Tuple[float, str, Customer]]:
        """Get the next departure event."""
        if self.in_service is not None:
            return (self.service_end_time, 'departure', self.in_service)
        return None
    
    def process_departure(self, current_time: float) -> Optional[Customer]:
        """Process a departure from the queue."""
        self.update_size_metrics(current_time)
        
        if self.in_service is None:
            return None
        
        # Complete service for current customer
        departing_customer = self.in_service
        departing_customer.departure_times[self.component_id] = current_time
        self.total_departures += 1
        self.customers_served += 1
        self.current_size -= 1
        
        # Calculate metrics
        queue_time = current_time - departing_customer.queue_entry_times[self.component_id]
        self.total_queue_time += queue_time
        
        # Move next customer to service
        if self.queue:
            self.in_service = self.queue.popleft()
            service_time = self.service_time_distribution()
            self.service_end_time = current_time + service_time
        else:
            self.in_service = None
            self.service_end_time = np.inf
        
        return departing_customer
    
    def average_queue_time(self) -> float:
        """Calculate average time spent in queue."""
        if self.customers_served > 0:
            return self.total_queue_time / self.customers_served
        return 0.0
    
    def utilization(self) -> float:
        """Calculate server utilization."""
        if self.current_time > 0:
            # Utilization is the fraction of time the server is busy
            busy_time = self.customers_served * self.average_queue_time()
            return min(busy_time / self.current_time, 1.0)
        return 0.0
