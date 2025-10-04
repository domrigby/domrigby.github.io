# RL Grokking Recipe- How Can We Enable LLMs to Solve Previously Unsolvable Tasks 

Date read: 3rd October

[ArXiv link](https://accessible-dragon-75f.notion.site/RL-Grokking-Recipe-How-Can-We-Enable-LLMs-to-Solve-Previously-Unsolvable-Tasks-with-RL-100a1714e6778062bae5eafad8e7677d)

## Key Points
* Problems:
	* How can algorithms learn in scenarios with zero reward? Gradients are minimal as reward landscape is flat. There is nothing to allow the model to differentiate between good and bad actions.
	* Model tend to not learn anything new during RLFT... pass@1 goes up, but pass@(high k) does not. This means the model is just increasing the probability of picking knowledge, reasoning paths it already had rather than learning new ones.

* Aim of paper: 
	* Increase performance in tasks at which pass@k is 0

* How?
	* **Two phase reward schedule**: dense and then binary for completing the task.
	* Binary: pass or fail reward
	* Dense: coding agent gets rewarded for passing unit tests. This causes the agent to over focus on easy unit tests, but the key insight it that is that in doing the total pass rate for the whole problem tends to go **just above 0**... reward signal!
	* Once the model begins to pass the pass the test more than 0% of the time it switches to binary reward

* Issue: 
	* Have to be able to come up with a dense reward which will get pass rate above 0.
	* Dense reward -> human bias