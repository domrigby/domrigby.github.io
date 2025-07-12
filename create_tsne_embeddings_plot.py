import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. your existing data
# 1. Dictionary of papers → one-sentence descriptions
papers = {
    "Absolute Zero Reasoner":
        "Introduces a reinforcement-learning based LLM agent that learns to chain deductive reasoning to solve zero-shot logical puzzles.",
    "Pi0.5":
        "Presents a framework combining vision-language alignment with pre-trained transformers to enable precise robotic manipulation in VLA tasks.",
    "TD-MPC2":
        "Extends model-predictive control with temporal-difference learning to improve sample-efficient control in non-LLM reinforcement settings.",
    "JointPPO":
        "Develops a multi-agent PPO algorithm that jointly optimizes policies for collaborative tasks with shared rewards.",
    "Synthetic Experience Replay":
        "Proposes generating synthetic trajectories to augment replay buffers and accelerate off-policy reinforcement learning.",
    "Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment":
        "Implements a turn-level credit assignment scheme to reinforce chain-of-thought consistency in multi-turn LLM interactions.",
    "Ultimate Guide to Supervised Fine-Tuning":
        "Surveys best practices and pitfalls of supervised fine-tuning for large language models, from data prep to hyperparameter selection.",
    "Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations (Demo Augmented RL)":
        "Combines demonstration-augmented off-policy RL with dexterous manipulation benchmarks to achieve robust robot grasping and manipulation.",
    "Spurious Rewards: Rethinking Training Signals in RLVR":
        "Analyzes how misleading reward functions degrade LLM-based RL and proposes corrected signal designs for more reliable training.",
    "ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models":
        "Shows that extended RL fine-tuning sessions unlock deeper multi-hop reasoning capabilities in LLMs.",
    "JaxMARL: Multi-Agent RL Environments and Algorithms in JAX":
        "Introduces a high-performance JAX-based suite of multi-agent environments and standardized MARL baselines.",
    "Illusion of Thinking: Understanding Strengths and Limitations of Large Reasoning Models (LRMs)":
        "Critically examines where chain-of-thought LLMs succeed or fail, using targeted benchmarks to reveal over-claimed reasoning prowess.",
    "CHIRPs: Change-Induced Regret Proxy Metrics for Lifelong Reinforcement Learning":
        "Defines proxy regret metrics based on environment shifts to evaluate lifelong RL agents’ adaptability.",
    "Enhanced POET: Open-Ended Reinforcement Learning through Unbounded Invention of Learning Challenges and their Solutions":
        "Extends POET with new generators and solvers to foster unbounded curriculum discovery in open-ended RL.",
    "Reinforcement Pre-Training":
        "Explores pre-training LLMs with RL objectives over synthetic tasks to prime them for downstream instruction following.",
    "Reinforcement Learning Teachers of Test Time Scaling":
        "Trains RL-based ‘teacher’ policies to adaptively scale LLM computation at inference for better quality–speed trade-offs.",
    "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning":
        "Shows that focusing RL updates on infrequent but informative tokens yields stronger chain-of-thought improvements.",
    "Writing-Zero: Bridge the Gap Between Non-Verifiable Tasks and Verifiable Rewards":
        "Designs a proxy reward model to supervise LLMs on tasks where ground-truth feedback is unavailable.",
    "Play to Generalize: Learning to Reason Through Game Play":
        "Uses self-play in simple games to teach LLMs general logical reasoning patterns transferable to novel problems.",
    "mimic-one: a Scalable Model Recipe for General Purpose Robot Dexterity":
        "Combines large-scale behavior cloning with modular policy fine-tuning to achieve versatile robot skills.",
    "Prioritised Level Replay":
        "Adapts prioritized experience replay to curriculum-driven levels, improving exploration in procedurally generated tasks.",
    "Self-Adapting Language Models":
        "Implements an online RL loop allowing LLMs to adjust their internal reward weights based on user feedback.",
    "Agents of Change: Self-Evolving LLM Agents for Strategic Planning":
        "Presents a framework for LLM agents that iteratively refine their own objectives via simulated rollout analysis.",
    "KINETIX: INVESTIGATING THE TRAINING OF GENERAL AGENTS THROUGH OPEN-ENDED PHYSICS-BASED CONTROL TASKS":
        "Benchmarks open-ended control in physics simulators to assess generality of RL agents across diverse tasks.",
    "Superintelligence From First Principles (blog post)":
        "Argues from first principles why open-ended exploration is key to emergent superintelligent capabilities.",
    "Automatic Curriculum Learning for Driving Scenarios: Towards Robust and Efficient Reinforcement Learning":
        "Uses automated curriculum generation to sequence increasingly difficult driving simulations for autonomous agents.",
    "How Visual Representations Map to Language Feature Space in Multimodal LLMs":
        "Probes alignment between visual encoder outputs and language embeddings in multimodal transformer architectures.",
    "Multi-Agent Training for Pommerman: Curriculum Learning and Population-based Self-Play Approach":
        "Combines population-based self-play with curriculum learning to train robust agents in the Pommerman environment.",
    "Automatic Curriculum Design for Zero-Shot Human AI Coordination":
        "Generates task curricula that maximize zero-shot coordination performance between humans and AI partners.",
    "OMNI-EPIC: Open-Endedness Via Models of Human Notions of Interestingness With Environments Programmed In Code":
        "Incorporates human curiosity models into environment generators to drive open-ended agent discovery.",
    "Self-Supervised Video Models Enable Understanding, Prediction and Planning (V-JEPA)":
        "Introduces a self-supervised video encoder that learns spatio-temporal features for downstream planning tasks.",
    "Open-Endedness is Essential for Artificial Superhuman Intelligence":
        "Makes the case that open-ended exploration and skill invention are prerequisites for achieving superhuman AI.",
    "SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning":
        "Demonstrates that self-play on adversarial games induces richer multi-turn reasoning in LLM-based agents.",
    "Training extremely large neural networks across thousands of GPUs by Jeremy Jordan":
        "Details system-level optimizations and communication strategies for scaling transformer training to >10k GPUs.",
    "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures":
        "Presents IMPALA, which decouples actors and learners with importance weighting to scale deep RL to many CPUs/GPUs.",
    "TorchBeast: A PyTorch Platform for Distributed RL":
        "Provides a PyTorch re-implementation of the SEED RL framework for high-throughput distributed training.",
    "Distributed PPO Blog Post":
        "Describes best practices for implementing and tuning distributed PPO at scale in real-world settings.",
    "Reinforcement Learning with Docker":
        "Shows how containerization with Docker can standardize RL experiments and simplify dependency management.",
    "FinCoT: Grounding Chain-of-Thought in Expert Financial Reasoning":
        "Adapts chain-of-thought prompting to financial domains by fine-tuning on expert annotated reasoning traces.",
}

titles, descs = list(papers), list(papers.values())

# 2. embed & project
model = SentenceTransformer('all-MiniLM-L6-v2')
emb = model.encode(descs, convert_to_numpy=True, show_progress_bar=False)
emb2 = TSNE(n_components=2, random_state=42, perplexity=15).fit_transform(emb)

# 3. cluster
labels = KMeans(n_clusters=6, random_state=42).fit_predict(emb2)

# 4. build a DataFrame
df = pd.DataFrame({
    'x': emb2[:,0], 'y': emb2[:,1],
    'cluster': labels.astype(str),
    'title': titles
})

# 5. plotly express!
fig = px.scatter(
    df, x='x', y='y',
    color='cluster',               # color‐by cluster
    hover_name='title',            # show paper title on hover
    labels={'x':'t-SNE dim 1','y':'t-SNE dim 2'},
    title="t-SNE of Paper Descriptions",
    size_max = 15
)
fig.update_layout(template='plotly_white', title_x=0.5)

# 6. dump as a standalone HTML
fig.write_html("tsne_papers.html", include_plotlyjs='cdn')
