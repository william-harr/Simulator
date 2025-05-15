# Simulator
Continuous Time Discrete Event Simulator

# Requirements
Python 3.8+
NumPy
SciPy
Pandas
Matplotlib
Seaborn

# Project Structure
queueing_system/
├── core/                           # Core components
│   ├── base.py                     # Base classes (Component, Customer, SystemMetrics)
│   ├── queue.py                    # Standard FIFO queue
│   ├── multiserver_queue.py        # Multi-server queue
│   └── pool.py                     # Pool with flexible selection
├── system/                         # System coordination
│   └── queueing_system.py          # Main simulation engine
├── distributions/                  # Random variable generators
│   └── random_variables.py         # Distribution functions
├── visualization/                  # Plotting and analysis
│   └── plotting.py                 # Visualization tools
├── scripts/                        # Utility scripts
│   ├── run_simulation.py           # CLI interface
│   ├── fit_distributions.py        # Distribution fitting tool
│   └── chick_fil_a_combinations.py # Run all distribution combinations
├── examples/                       # Example models
│   └── chick_fil_a_model.py        # Chick-fil-A drive-through model
└── main.py                         # Main entry point

Baseline model for chick-fil-a is in 'examples'. A script that runs all combinations of hypothetical distributions (10 times each) is in scripts. Hypothetical distributions came from StatFit and we selected based on the fit from StatFit and the underlying assumptions about the distribution as it relates to the senario. 

From the Command Line:
python scripts/chick_fil_a_combinations.py

We allow general simulation models by creating basic components. These include a queue, a multiserver queue, and a pool. 

A queue is a standard first-in first-out. The 'in_service' parameter determines if the service is considered separately from the line. 
For chick-fil-a specifically, the drive through had a distinct service element, whereas the food service does not. 

The pool differs from the queue because anyone in the queue can be selected at any time. 
In a homogeneous pool, all members have the same chance of getting selected. 
In a heterogeneous pool, different members may have a different probability of being selected. This may be based on them, how long they've been in the pool, or some other factor.
The most accurate model of chick-fil-a would have a delay (serviceless queue) followed by a heterogenerous pool where wait times depend on order size and time in the pool. Given our data collection we would not have been able to make any meaningful change with this, so we let it be. 

Distinct components can be brought together into a complete system with the Queueing_System class in system/queueing_system.py.
Outputs of components can be used as inputs for later components using the routing matrix. This is how we connect multiple queues, delays, pools etc. We use it for our Chick_Fil_A model. 

## Metrics and Data Capture ##
System-Level Metrics

Total Customers Created: Number of customers generated
Total Customers Completed: Number of customers that exited the system
Average System Size: Time-weighted average number of customers in system
Average System Time: Average time customers spend in the system
System Throughput: Customers completed per unit time
Current Customers in System: Real-time count

Component-Level Metrics
Queue Metrics

Total Arrivals: Customers that entered the queue
Total Departures: Customers that left the queue
Average Queue Size: Time-weighted average queue length
Current Queue Size: Real-time queue length
Average Queue Time: Average time spent waiting and being served
Utilization: Fraction of time server is busy

MultiServerQueue Metrics

Server Utilization: Average utilization across all servers
Average Service Time: Mean time to serve a customer
Server Busy Time: Total time servers were occupied
All standard queue metrics

Pool Metrics

Average Wait Time: Mean time customers spend in pool
All standard component metrics

Customer-Level Tracking
Each customer object tracks:

Customer ID: Unique identifier
Creation Time: When customer entered system
Order Size: Customer-specific attribute (optional)
Priority: For priority-based selection (optional)
Arrival Times: Dictionary of arrival times at each component
Departure Times: Dictionary of departure times from each component
Queue Entry Times: When customer joined each queue


## Component Types ##
Queue
Standard first-in-first-out queue with service at the front.

Parameters: service_time_distribution, capacity
Behavior: Customers wait in line and are served when they reach the front

MultiServerQueue
Queue with multiple parallel servers.

Parameters: service_time_distribution, num_servers, capacity
Behavior: Customers are served by the first available server

Pool
Collection where any customer can be selected at any time.

Parameters: exit_time_distribution, capacity, selection_method
Selection Methods:

fcfs: First-come-first-served
priority: Based on customer priority
random: Random selection
