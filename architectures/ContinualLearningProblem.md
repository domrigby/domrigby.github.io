# The Continual Learning Problem - Jessy Lin

Date read: 28th October 2025

[Blog link](https://jessylin.com/2025/10/20/continual-learning/)

## Key Points
* Issue: how do we continually learn without destroying everything we learned in pre-training?
* It should be a key capability of AI systems that they can constantly get smarter and learn new things.

* Key challenges:
	1. **Generalisation**:
		* How can we learn lessons from new data that generalises to more new data?
		* We must convert facts to knowledge. E.g. we don't want to predict the next work from a new sentence... we want to extract its meaning. Paraphrasing can help this.	
		* Models should learn to self-supervise better using new data.
	
	2. **Forgetting / Integration**:
		* We want to update old knowledge as well as add new facts
		* Models must know what to get rid of as well as be able to add new info.
		* We want to find and only change the weights responsible for making a decision.

