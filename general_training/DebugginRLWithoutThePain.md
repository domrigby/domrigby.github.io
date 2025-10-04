# Debugging RL, Without the Agonising Pain

Date read: 4th October 2025

[Blog link](https://andyljones.com/posts/rl-debugging.html?utm_source=chatgpt.com)

## Key Points
* **Debugging is hard!**:
	* Errors propagate through the system and don't always cause crashes.
	* Implementation quality does not correlate with results... even buggy code can learn something
	* Interfaces can be very large

* Recommended practices:
	* **Simple tests**. Make sure they:
		1. Do not rely on seeds. This will make debugging a nightmare
		2. Run quickly (prefferably seconds)
		3. Split the code as much as possible (divide and conquer debugging). This will make it far easier to binary search and work out where the error is
		4. Isolate parts where bugs are more likely

	* **Always chase odd behaviours**: it is never a problem for another day and will cause huge headaches later on if you leave it. 
	* **Reward is in the [-1, 1] range**: it is far easier to do this by manual constants rather than adaptive reward schemes
	* **As large a batch size as possible**:
		* In odd environments it is possible to get full batches of odd experiences... larger batches make this less likely.
	* **Start small with networks**
	* **Vectorised environments**: record the reset step number... should be evenly distributed. If not then it may mean environments are correlated.
	* **Assume you have a bug before you change hyperparameters**: i.e. have a high threshold for accepting the code is correct.

* ** Probe environments**: 
	* Blog includes a really good a series of simple and very fast environments which running in sequence will help isolate bugs.

* **Probe agents**:
	* Train agents with extra information or an oracle... if these can't learn then nothing else will.
	* This relies on having customisable interfaces

* **Diagnostic stats**:
	* Relative entropy:
		* Ratio of max entropy to current entropy of policy
		* Too high: not learning
		* Too low: mode collapse -> add entropy bonus
		* High variance: LR too high

	* KL-Divergence:
		* Measures difference between current and reference policy.
		* Very low: not much change, you can turn up LR
		* Growing: constantly seeing staler data... check sampling

	* Residual variance: 
		* (Q_target - Q) / \sigma(Q_target)
		* Should start close to 1.
		* If it does not decrease... not learning

	* Terminal correlation:
		* Correlation between predicted value or terminal state and reward at terminal state
		* Should go to 1... if it does not check rewards and value

	* Plot distributions of value, reward, advantage etc

	* GPU stats etc.








