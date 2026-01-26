import numpy as np
import copy


class Storage(object):

    def __init__(self, env, population, num_past, step):
        """
        Initialize storage for collecting agent trajectories.
        
        Args:
            env: GridWorld environment
            population: List of agents
            num_past: Number of past episodes to collect
            step: Number of steps per episode
        """
        self.env = env
        self.population = population
        self.num_past = num_past
        self.step = step
        
        # Storage arrays will be initialized in extract() based on split
        self.past_trajectories = None
        self.current_state = None
        self.target_action = None
        self.dones = None
        self.action_count = None

    def extract(self, custom_env=-100, split=None):
        """
        Extract trajectories from agents.
        
        Args:
            custom_env: Custom environment configuration (default: -100)
            split: Optional split mode for 80/20 train/eval split:
                   - 'train': agents where index % 5 != 0 (80%)
                   - 'eval': agents where index % 5 == 0 (20%)
                   - None: use all agents
        
        Returns:
            Dictionary with episodes, curr_state, target_action, dones
        """
        # Determine which agents to process based on split
        if split == 'train':
            # Training: agents where index % 5 != 0 (80%)
            agent_indices = [i for i in range(len(self.population)) if i % 5 != 0]
        elif split == 'eval':
            # Evaluation: agents where index % 5 == 0 (20%)
            agent_indices = [i for i in range(len(self.population)) if i % 5 == 0]
        else:
            # No split: use all agents
            agent_indices = list(range(len(self.population)))
        
        num_agents = len(agent_indices)
        
        # Initialize storage arrays based on number of agents to process
        self.past_trajectories = np.zeros([num_agents, self.num_past, self.step, 
                                           self.env.height, self.env.width, 11])
        self.current_state = np.zeros([num_agents, self.env.height, self.env.width, 6])
        self.target_action = np.zeros([num_agents, 5])
        self.dones = np.zeros([num_agents, self.num_past, self.step, 1])
        self.action_count = np.zeros([num_agents, 5])
        
        for storage_idx, agent_index in enumerate(agent_indices):
            agent = self.population[agent_index]

            for past_epi in range(self.num_past):
                if np.sum(custom_env) > 0:
                    obs = self.env.reset(custom=custom_env[storage_idx])
                else:
                    obs = self.env.reset()

                # gathering past trajectories
                for step in range(self.env.epi_max_step):
                    action = agent.act(obs)
                    self.action_count[storage_idx, action] += 1
                    spatial_concat_action = np.zeros((self.env.height, self.env.width, 5))
                    spatial_concat_action[:, :, action] = 1

                    obs_concat = np.concatenate([obs, spatial_concat_action], axis=-1)
                    self.past_trajectories[storage_idx, past_epi, step] = obs_concat
                    self.dones[storage_idx, past_epi, step] = 1

                    obs, reward, done, _ = self.env.step(action)
                    if done:
                        break

            # gathering current_state
            for _ in range(1):
                curr_obs = self.env.reset()
                target_action = agent.act(curr_obs)
                self.current_state[storage_idx] = curr_obs
                self.target_action[storage_idx, target_action] = 1
            print('Agent {} (index {}) make complete!'.format(storage_idx, agent_index))

        return dict(episodes=self.past_trajectories,
                    curr_state=self.current_state,
                    target_action=self.target_action,
                    dones=self.dones)

    def reset(self):
        self.past_trajectories = np.zeros(self.past_trajectories.shape)
        self.current_state = np.zeros(self.current_state.shape)
        self.target_action = np.zeros(self.target_action.shape)
        self.dones = np.zeros(self.dones.shape)
        self.action_count = np.zeros(self.action_count.shape)

    def get_most_act(self):
        action_count = copy.deepcopy(self.action_count)
        row_sums = np.sum(action_count, axis=-1)
        # Avoid division by zero: if an agent has no actions, set sum to 1
        row_sums = np.where(row_sums == 0, 1, row_sums)
        action_count /= np.reshape(row_sums, (-1, 1))

        return np.argmax(action_count, axis=-1), np.max(action_count, axis=-1)

if __name__ == '__main__':
    import sys
    import os
    
    # Add project root to path so imports work when running directly
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    from environment.env import GridWorldEnv
    from utils import utils
    
    # Paper specifications:
    # 1. For each species S(α), train a single ToMnet
    # 2. For each agent: Npast ~ U{0, 10} (variable number of past episodes)
    # 3. Each episode length = 1 (single state-action pair)
    # 4. When no past episodes: echar = 0
    
    # Experiment 1 configuration (from config.py)
    num_agents = 100
    alpha = 0.01
    move_penalty = -0.01
    
    # Check config - paper says Npast ~ U{0, 10}, but current implementation uses fixed num_past
    from experiment1.config import get_configs
    exp_kwargs, env_kwargs, model_kwargs, agent_kwargs = get_configs(1)  # num_exp=1
    num_past = exp_kwargs['num_past']  # This is currently fixed, not sampled per agent
    num_step = exp_kwargs['num_step']
    
    env = GridWorldEnv(env_kwargs)
    
    print(f"\n{'='*70}")
    print("PAPER SPECIFICATION CHECK:")
    print(f"{'='*70}")
    print(f"\n1. Species S(α):")
    print(f"   Using single alpha value: {alpha}")
    print(f"   Training single ToMnet per species")
    
    print(f"\n2. Past Episodes (Npast):")
    print(f"   PAPER: Npast ~ U{{0, 10}} (variable per agent)")
    print(f"   CURRENT: Fixed num_past = {num_past} for all agents")
    
    print(f"\n3. Episode Length:")
    print(f"   PAPER: Each episode = 1 (single state-action pair)")
    print(f"   CURRENT: env.epi_max_step = {env.epi_max_step}")
    print(f"   CURRENT: num_step = {num_step}")
    if env.epi_max_step == 1 and num_step == 1:
        print(f"   MATCHES PAPER SPECIFICATION")
    else:
        print(f"   DOES NOT MATCH - should be 1")
    
    print(f"\n{'='*70}")
    print("DATA COLLECTION TEST:")
    print(f"{'='*70}")
    
    # Create population of random agents
    print(f"\nCreating {num_agents} random agents with alpha={alpha}...")
    population = utils.make_pool('random', move_penalty, alpha, num_agents)
    
    # Create storage
    print(f"Initializing storage with num_past={num_past}, step={num_step}...")
    storage = Storage(env, population, num_past, num_step)
    
    # Extract trajectories
    print(f"\nExtracting trajectories from {num_agents} agents...")
    data_dict = storage.extract()
    
    # Verify data structure
    print(f"\n{'='*70}")
    print("DATA STRUCTURE VERIFICATION:")
    print(f"{'='*70}")
    
    episodes = data_dict['episodes']
    current_state = data_dict['curr_state']
    target_action = data_dict['target_action']
    
    print(f"\nShapes:")
    print(f"  past_trajectories: {episodes.shape}")
    print(f"    → [num_agents={episodes.shape[0]}, num_past={episodes.shape[1]}, "
          f"num_step={episodes.shape[2]}, height={episodes.shape[3]}, "
          f"width={episodes.shape[4]}, channels={episodes.shape[5]}]")
    print(f"  current_state: {current_state.shape}")
    print(f"  target_action: {target_action.shape}")
    
    # Check episode length
    print(f"\nEpisode Length Check:")
    print(f"  Each past episode should contain exactly 1 state-action pair")
    print(f"  num_step dimension: {episodes.shape[2]}")
    print(f"  env.epi_max_step: {env.epi_max_step}")
    if episodes.shape[2] == 1 and env.epi_max_step == 1:
        print(f"  CORRECT: Each episode is length 1")
    else:
        print(f"  INCORRECT: Episode length should be 1")
    
    # Check trajectory content
    print(f"\nTrajectory Content Check:")
    sample_agent = 0
    sample_episode = 0
    traj = episodes[sample_agent, sample_episode, 0]  # [height, width, 11]
    print(f"  Sample trajectory shape: {traj.shape}")
    print(f"  Channels: 0-5 (observation), 6-10 (action encoding)")
    
    # Count non-zero trajectories
    non_zero_trajs = 0
    for agent_idx in range(episodes.shape[0]):
        for epi_idx in range(episodes.shape[1]):
            # Check if trajectory has agent position
            if np.any(episodes[agent_idx, epi_idx, 0, :, :, 5] == 1):
                non_zero_trajs += 1
    
    print(f"\nTrajectory Statistics:")
    print(f"  Total trajectory slots: {episodes.shape[0] * episodes.shape[1]}")
    print(f"  Non-zero trajectories: {non_zero_trajs}")
    print(f"  Zero trajectories (empty): {episodes.shape[0] * episodes.shape[1] - non_zero_trajs}")
    
    # Action distribution
    total_actions = storage.action_count.sum(axis=1)
    total_actions = np.where(total_actions == 0, 1, total_actions)
    action_proportions = storage.action_count / total_actions.reshape(-1, 1)
    action_names = ['Stay', 'Down', 'Right', 'Up', 'Left']
    
    print(f"\nAction Distribution (across all agents):")
    for i, name in enumerate(action_names):
        mean_prop = action_proportions[:, i].mean()
        print(f"  {name:8s}: {mean_prop:.4f}")