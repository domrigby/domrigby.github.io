# Communicative Reinforcement Learning Agents for Landmark Detection in Brain Images

Date read: 29th July 2025. Read due to interesting use of MARL.

[arXiv link](https://arxiv.org/abs/2008.08055)

## Key Points
* In large 3D brain scans it is important to find 'landmarks' in the brain (cortexes etc).
* Processing this in one go can be inefficient due to large amount of data and treating this like an RL problem in which
an agent searches for the landmarks has shown better performance.
* This paper poses a **multi-agent communications method** for tackling this problem. 
* Deploys double DQN with **shared convolutional layers and concatenating features in the final fully connected layers**.
* Agents have a 45x45x45 box and decided which direction they want to move in (preferably towards a landmark). This 
starts off with 3mm moves but moves down to smaller moves once an agent starts oscillating.