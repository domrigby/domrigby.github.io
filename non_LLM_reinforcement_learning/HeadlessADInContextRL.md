# In-Context Reinforcement Learning for Variable Action Spaces

Date read: 1st September 2025

[arXiv link](https://arxiv.org/pdf/2312.13327)

## Key Points
* Introduces HeadlessAD, an in context reinforcement learning method which takes sequences of states, actions and rewards and then continues to
guess the next optimal action.
* HeadlessAD works on **varying sized discrete action spaces**
* This is achieved by operating in an abstract action embedding space to make the model action size invariant.
* Action embedding is **randomly and orthonormally initialised** for each sequence to prevent the model utilising inbuilt knowledge of the game 
and forcing it to learn in context.
* Contrastive loss is used to train the model to predict the embeddings.
* They finetune TinyLlama LLM