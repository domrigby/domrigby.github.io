# AlphaProof Paper - Julian Schrittwieser

Date read: 13th November 2025

[Blog link]()

## Key Points
* Explains the design of AlphaProof, an algorithm which can output proofs of Internation Math Olympiad problems
* How?
	* Uses **Lean**: a language for writing maths proofs which can be automatically verified	
	* Lean proofs are made up of a set of **proof 'tactics'** e.g. proof by induction
	* AlphaProof therefore treats proofs like a game in which you choose tactics to try and solve it.
	
* Architecture:
	* Utilise AlphaZero like MCTS 
	* Each node represent a **partial solution** (uncompleted lean states)
	* Encoder-decoder architecture	
	* Added special **product node**: some tactics generate a set of sub proofs you must complete (e.g. if true for 0 and any real number). Backpropagates reward of hardest sub proof (this defines how hard whole problem is)

* Training:
	* Only pre-trained on maths and code, not large scale web data
	* Smaller model
	* Wanted to maximise signal: performed next token prediction + span reconstruction
	* Trained on mathlib: 200k proofs
	* Also trained a model to convert natural language problems to lean and generated more problem
	* Started with a small number of rollouts and gradually increased it (only solve easy problems at start, don't waste time on impossible ones)

* Test-Time RL:
	* Use LLM to reword the quest thousands of time
	* Do RL live to get better and better answers to the problem