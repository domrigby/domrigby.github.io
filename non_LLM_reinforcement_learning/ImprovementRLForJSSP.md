# Deep Reinforcement Learning Guided Improvement Heuristic For Job Shop Scheduling

Date read: 12th October 2025

[Paper link]()

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