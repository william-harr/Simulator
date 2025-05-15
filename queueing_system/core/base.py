"""Base classes for the queueing system."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class Customer:
    """Represents a customer flowing through the queueing system."""
    customer_id: int
    creation_time: float
    order_size: float = 1.0
    priority: float = 0.0  # Lower values = higher priority
    arrival_times: Dict[str, float] = field(default_factory=dict)
    departure_times: Dict[str, float] = field(default_factory=dict)
    queue_entry_times: Dict[str, float] = field(default_factory=dict)
    
    def system_time(self, component_id: str) -> float:
        """Calculate total time spent in a component."""
        if component_id in self.departure_times and component_id in self.arrival_times:
            return self.departure_times[component_id] - self.arrival_times[component_id]
        return 0.0
    
    def queue_time(self, component_id: str) -> float:
        """Calculate time spent waiting in queue (before service)."""
        if component_id in self.queue_entry_times and component_id in self.departure_times:
            service_start = self.departure_times[component_id] - self.queue_entry_times[component_id]
            return service_start
        return 0.0


class Component(ABC):
    """Base class for all queueing system components."""
    
    def __init__(self, component_id: str):
        self.component_id = component_id
        self.current_time = 0.0
        self.total_arrivals = 0
        self.total_departures = 0
        self.current_size = 0
        
        # Metrics tracking
        self.size_time_product = 0.0  # Integral of size over time
        self.last_update_time = 0.0
        
    def update_size_metrics(self, new_time: float):
        """Update size-based metrics before changing system state."""
        time_delta = new_time - self.last_update_time
        self.size_time_product += self.current_size * time_delta
        self.last_update_time = new_time
        self.current_time = new_time
    
    def average_size(self) -> float:
        """Calculate average number of customers in component."""
        if self.current_time > 0:
            return self.size_time_product / self.current_time
        return 0.0
    
    @abstractmethod
    def process_arrival(self, customer: Customer, current_time: float) -> bool:
        """
        Process an arriving customer.
        Returns True if customer was accepted, False if rejected (due to capacity).
        """
        pass
    
    @abstractmethod
    def get_next_event(self) -> Optional[Tuple[float, str, Customer]]:
        """
        Get the next event for this component.
        Returns (event_time, event_type, customer) or None if no events.
        """
        pass


@dataclass
class SystemMetrics:
    """Tracks system-wide metrics."""
    total_customers_created: int = 0
    total_customers_completed: int = 0
    total_system_time: float = 0.0
    customers_in_system: int = 0
    system_time_product: float = 0.0
    last_update_time: float = 0.0
    
    def update_metrics(self, new_time: float):
        """Update time-based metrics."""
        time_delta = new_time - self.last_update_time
        self.system_time_product += self.customers_in_system * time_delta
        self.last_update_time = new_time
    
    def average_system_size(self, current_time: float) -> float:
        """Calculate average number of customers in system."""
        if current_time > 0:
            return self.system_time_product / current_time
        return 0.0
    
    def average_system_time(self) -> float:
        """Calculate average time customers spend in system."""
        if self.total_customers_completed > 0:
            return self.total_system_time / self.total_customers_completed
        return 0.0
    
    def throughput(self, current_time: float) -> float:
        """Calculate system throughput (customers/time)."""
        if current_time > 0:
            return self.total_customers_completed / current_time
        return 0.0