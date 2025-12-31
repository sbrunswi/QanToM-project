# Experiment 1: Random Agents

This experiment implements Section 3.1 of the Machine Theory of Mind paper (Rabinowitz et al., ICML 2018). It trains a ToMnet (Theory of Mind neural network) to predict the future actions of random agents in a gridworld environment.

## Overview

The ToMnet learns to model agents with random policies, where each agent's action distribution is sampled from a Dirichlet distribution with parameter α. The network observes past trajectories of agent behavior and uses this to predict their future actions in novel states.

## Paper Reference

**Section 3.1: Random Agents** from [Machine Theory of Mind (arXiv:1802.07740)](https://arxiv.org/pdf/1802.07740)

Key points from the paper:
- Agents have policies sampled from Dirichlet(α) distributions
- Each episode consists of a single state-action pair (episode length = 1)
- For each agent, N_past ~ U{0, 10} past episodes are observed (variable per agent)
- The ToMnet learns to predict action probabilities from past behavioral observations

## Architecture

The ToMnet consists of two main components:

1. **CharNet (Character Network)**: Processes past trajectories to extract character embeddings (e_char)
   - Convolutional layers + LSTM to process trajectory sequences
   - Outputs a 2D character embedding summarizing agent behavior

2. **PredNet (Prediction Network)**: Predicts future actions
   - Combines character embeddings with current state observations
   - Outputs probability distribution over 5 actions: [Stay, Down, Right, Up, Left]

## Data Format

### Trajectories
- **Past trajectories**: Shape `[num_agents, num_past, num_step, height, width, 11]`
  - Each trajectory is a state-action pair: `(11×11×6) observation + (11×11×5) spatialized action = (11×11×11)`
  - Channels 0-5: Observation (walls, objects, agent)
  - Channels 6-10: One-hot action encoding (spatialized)

- **Current state**: Shape `[num_agents, height, width, 6]`
  - Observation at prediction time

- **Target action**: Shape `[num_agents, 5]`
  - One-hot encoded true action taken by agent

### Pre-processing
As described in the paper, actions are "spatialised" by tiling the 5-dimensional action vector over the 11×11 grid, then concatenated with the state to form a single tensor.

## Training Configuration

### Paper Specifications
- **Optimizer**: Adam
- **Learning rate**: 10⁻⁴
- **Batch size**: 16
- **Training duration**: 40k minibatches (for random agents)
- **Grid size**: 11×11
- **Episode length**: 1 (single state-action pair per episode)

### Current Implementation
- Default batch size: 16
- Default learning rate: 1e-4
- Training by epochs (not minibatches)
- To match 40k minibatches: `num_epochs ≈ 40000 / (num_agents / batch_size)`

### Example

```bash
python main.py --main_exp 1 --sub_exp 1 --num_agent 1000 --batch_size 16 --lr 1e-4 --num_epoch 640 --alpha 0.01
```

## Implementation Details

### Key Components

1. **`config.py`**: Configuration settings
   - Sets `num_past = sub_exp` (number of past episodes)
   - Configures environment (11×11 grid, exp=1)
   - Sets model parameters

2. **`model.py`**: ToMnet architecture
   - `CharNet`: Character embedding network
   - `PredNet`: Action prediction network
   - Training and evaluation methods

3. **`store_trajectories.py`**: Data collection
   - `Storage` class: Extracts trajectories from agent population
   - Spatializes actions and concatenates with observations
   - Handles variable number of past episodes

4. **`experiment.py`**: Main experiment loop
   - Creates agent population
   - Collects training/evaluation data
   - Trains ToMnet
   - Generates visualizations

### Data Collection Process

1. Create population of random agents (each with Dirichlet(α) policy)
2. For each agent:
   - Generate `num_past` past episodes (each = 1 state-action pair)
   - Generate current state and target action
3. Store as tensors for training

### Training Process

1. Forward pass: ToMnet processes past trajectories → character embedding
2. Combine embedding with current state → action prediction
3. Loss: KL divergence between predicted and true action distributions
4. Backpropagation and optimization

## Outputs

The experiment generates:
- Model checkpoints (saved periodically)
- Training/evaluation metrics (loss, accuracy)
- Visualizations:
  - Past trajectories
  - Current states
  - Predicted action distributions
  - Character embeddings (t-SNE plots)

## Differences from Paper

1. **N_past**: Paper uses N_past ~ U{0, 10} (variable per agent), implementation uses fixed `num_past` for all agents
2. **Training duration**: Uses epochs instead of minibatches
3. For proper generalization, consider splitting train/eval by agents rather than using same population


## References

Rabinowitz, N. C., et al. (2018). Machine Theory of Mind. *ICML 2018*. [arXiv:1802.07740](https://arxiv.org/pdf/1802.07740)

