"""Multi-server queue component implementation."""

from collections import deque
from typing import Callable, Optional, Tuple, List
import numpy as np
import heapq

from core.base import Component, Customer


class MultiServerQueue(Component):
    """Queue with multiple servers in parallel."""
    
    def __init__(self,
                 component_id: str,
                 service_time_distribution: Callable[[], float],
                 num_servers: int = 1,
                 capacity: float = np.inf):
        super().__init__(component_id)
        self.service_time_distribution = service_time_distribution
        self.num_servers = num_servers
        self.capacity = capacity
        self.queue = deque()
        
        # Track which customers are being served by which server
        self.servers = [None] * num_servers  # None means server is idle
        self.server_end_times = [np.inf] * num_servers
        
        # Additional metrics
        self.total_queue_time = 0.0
        self.total_service_time = 0.0  # Track actual service time
        self.customers_served = 0
        self.server_busy_time = 0.0  # Total busy time across all servers
        
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
        
        # Find an idle server
        idle_server = self._find_idle_server()
        
        if idle_server is not None:
            # Assign customer to idle server
            self.servers[idle_server] = customer
            service_time = self.service_time_distribution()
            self.server_end_times[idle_server] = current_time + service_time
            self.total_service_time += service_time
        else:
            # All servers busy, add to queue
            self.queue.append(customer)
        
        self.current_size += 1
        return True
    
    def _find_idle_server(self) -> Optional[int]:
        """Find an idle server, return None if all busy."""
        for i, server in enumerate(self.servers):
            if server is None:
                return i
        return None
    
    def _find_next_completing_server(self) -> Optional[int]:
        """Find the server that will complete service next."""
        min_time = np.inf
        min_server = None
        
        for i, end_time in enumerate(self.server_end_times):
            if end_time < min_time:
                min_time = end_time
                min_server = i
        
        return min_server
    
    def get_next_event(self) -> Optional[Tuple[float, str, Customer]]:
        """Get the next departure event."""
        next_server = self._find_next_completing_server()
        
        if next_server is not None and self.servers[next_server] is not None:
            return (self.server_end_times[next_server], 'departure',
                   self.servers[next_server])
        
        return None
    
    def process_departure(self, current_time: float) -> Optional[Customer]:
        """Process a departure from the queue."""
        self.update_size_metrics(current_time)
        
        # Find which server is completing
        completing_server = self._find_next_completing_server()
        
        if completing_server is None or self.servers[completing_server] is None:
            return None
        
        # Track server busy time
        service_start_time = self.server_end_times[completing_server] - (
            self.server_end_times[completing_server] -
            self.servers[completing_server].queue_entry_times[self.component_id]
        )
        self.server_busy_time += (current_time - service_start_time)
        
        # Complete service for current customer
        departing_customer = self.servers[completing_server]
        departing_customer.departure_times[self.component_id] = current_time
        self.total_departures += 1
        self.customers_served += 1
        self.current_size -= 1
        
        # Calculate metrics
        queue_time = current_time - departing_customer.queue_entry_times[self.component_id]
        self.total_queue_time += queue_time
        
        # Check if there's a customer waiting in queue
        if self.queue:
            # Move next customer to this server
            next_customer = self.queue.popleft()
            self.servers[completing_server] = next_customer
            service_time = self.service_time_distribution()
            self.server_end_times[completing_server] = current_time + service_time
            self.total_service_time += service_time
        else:
            # Server becomes idle
            self.servers[completing_server] = None
            self.server_end_times[completing_server] = np.inf
        
        return departing_customer
    
    def average_queue_time(self) -> float:
        """Calculate average time spent in queue."""
        if self.customers_served > 0:
            return self.total_queue_time / self.customers_served
        return 0.0
    
    def average_service_time(self) -> float:
        """Calculate average service time."""
        if self.customers_served > 0:
            return self.total_service_time / self.customers_served
        return 0.0
    
    def server_utilization(self) -> float:
        """Calculate average server utilization."""
        if self.current_time > 0 and self.num_servers > 0:
            return min(self.server_busy_time / (self.current_time * self.num_servers), 1.0)
        return 0.0
    
    def update_size_metrics(self, new_time: float):
        """Update size-based metrics and server busy time."""
        time_delta = new_time - self.last_update_time
        
        # Count busy servers
        busy_servers = sum(1 for server in self.servers if server is not None)
        self.server_busy_time += busy_servers * time_delta
        
        # Call parent update
        super().update_size_metrics(new_time)
