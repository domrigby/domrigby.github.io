# On the Design of KL-Regularised Policy Gradient Algorithms for LLM Reasoning

Date read: 30th September 2025

[ArXiv Link](https://arxiv.org/pdf/2505.17508)

## Key Points
* Derives policy gradients for KL-regularised policies.
* KL-divergence types: imagine p(x) is the real distribution and q(x) is our models predicted distribution
	1. Forward KL: 
		* log(p(x)/q(x)): real dist on top, model dist below
		* MODE COVERING: must cover all non-zero areas of p(x) other wise KL will explode**
		* Why? If q(x) = 0 somewhere p(x) does not then KL -> inf
	2. **Backward KL**:
		* log(q(x)/p(x)): model dist on top, real dist below
		* **MODE SEEKING: only cares about areas in which q(x) has mass (0 otherwise).
		* Will collapse to a single mode if required
		* Why? If q(x) = 0 then it doesnt matter what p(x) equals.
	3. Normalise and unnormalised KL:
		* If KL is not normalised then apply a mass correction.

* Benefits of KL:
	* Control policy update size
	* Prevent large deviations
	* Prevent catastrophic forgetting.

* Methods:
	* Can be incorporated into reward as a punishment for high KL 

* Many types of estimator: k1, k2, k3 etc

* Proposed weight importance sampling weight for off-policy data when sampling KL divergence