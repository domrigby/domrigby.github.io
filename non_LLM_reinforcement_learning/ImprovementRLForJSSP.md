# Deep Reinforcement Learning Guided Improvement Heuristic For Job Shop Scheduling

Date read: 12th October 2025

[Paper link](https://arxiv.org/abs/2211.10936)

## Key Points

* Problem:
	* JSSP solvers tend to focus on construction heuristics
	* Construction heuristics: recurssively update partial solutions by adding actions
	* Issue: graphs struggle to encorporate partial solution information (e.g. current machine load and job status)
	* Therefore asks questions: **can we convert construction problem into a search one?** (don't have to model partial solutions)

* **Job Shop Scheduling Problem**:
	* We have J jobs and M machines
	* Each job requires a set of specific machines to be used in order and it differs per job.
	* Aim is to allocate jobs to machines to minimise the time taken complete all jobs

* Common method for representing this problem is a **disjunctive graph**:
	* Made up of parts:
		* Nodes: O_ji -> ith operation of job j
		* Conjunctive arcs: directed edges which connect job orders (must exist)
		* Disjunctive arcs: undirected edges which connect operations requiring the same machine. There is mutual exclusion, only one can be processed at a time. Eventually receive order 
	* Finding a solution = fixing disjunction directions
	* Resulting graph is a Directed Acyclic Graph (cycles would mean circular dependenies)
	* **Critical path**: longest path for a machine from start to end (defines our reward)
	* **Critical block**: string of operations on one machine on a critical path.
	* **Neighbourhood block**: identify critical blocks... the neighbourhood is potential changes we could make by swapping operations


* Traditional methods use a heuristic to selecta change withinn the neighbourhood (e.g. greedy) 

* Algorithm:
	* Learns via REINFORCE
	* State is current solution
	* Action is which operation to swap to improve the solution.
		* Selects Operation from OxO size matrix
	* Reward: time improvement per step
	* Architecture: Graph Attention 

* Representation. Separate graphs for:
	1. Precedence constraints (which operations must be processed before what)
	2. Current processing order
