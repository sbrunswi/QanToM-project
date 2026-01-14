from environment.env import GridWorldEnv
from new_experiment1 import model
from new_experiment1.store_trajectories import Storage
from new_experiment1.config import get_configs
from new_experiment1.trainer import train_epoch, eval_model

from utils.visualize import *
from utils import utils
from utils import dataset
from utils import writer

from torch.utils.data import DataLoader
import torch as tr
import torch.optim as optim
import numpy as np
import os


def visualize_results(ev_results, visualizer, most_act=None, count_act=None):
    """Helper function to visualize evaluation results.
    
    Args:
        ev_results: Dictionary from eval_model with visualization data
        visualizer: Visualizer instance
        most_act: Most common actions (optional)
        count_act: Action counts (optional)
    """
    past_traj = ev_results.get('past_traj', None)
    num_past = past_traj.shape[1] if past_traj is not None else 0
    
    for n in range(min(16, len(ev_results['curr_state']))):
        # Only visualize past trajectories if num_past > 0
        if num_past > 0 and past_traj is not None:
            # Extract agent positions and past actions from trajectory
            agent_xys = np.where(past_traj[n, 0, :, :, :, 5] == 1)
            # Extract past actions: channels 6-10 are action encodings
            env_height, env_width = past_traj[n, 0, 0].shape[0], past_traj[n, 0, 0].shape[1]
            _, past_actions = np.where(past_traj[n, 0, :, :, :, 6:].sum((1, 2)) == env_height * env_width)
            visualizer.get_past_traj(past_traj[n][0][0], agent_xys, past_actions, 0, sample_num=n)
        visualizer.get_curr_state(ev_results['curr_state'][n], 0, sample_num=n)
        visualizer.get_action(ev_results['pred_actions'][n], 0, sample_num=n)

    visualizer.get_action_char(ev_results['e_char'], most_act, count_act, 0)

def run_experiment(num_epoch, past, num_agent, batch_size, learning_rate,
                   experiment_folder, alpha, save_freq, train_dir='none', eval_dir='none', device=None):
    """
    Run the Theory of Mind experiment.
    
    Args:
        num_epoch: Number of training epochs
        past: Number of past episodes (num_past)
        num_agent: Number of agents in the population
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        experiment_folder: Folder to save experiment results
        alpha: Alpha parameter(s) for agent population
        save_freq: Frequency of checkpoint saving
        train_dir: Directory for training data (default: 'none' - generate on fly)
        eval_dir: Directory for eval data (default: 'none' - generate on fly)
        device: Device to run on ('cpu', 'cuda', 'mps')
    """
    exp_kwargs, env_kwargs, model_kwargs, agent_kwargs = get_configs(past)
    train_population = utils.make_pool('random', exp_kwargs['move_penalty'], alpha, num_agent)
    eval_population = utils.make_pool('random', exp_kwargs['move_penalty'], alpha, num_agent)
    env = GridWorldEnv(env_kwargs)
    model_kwargs['device'] = device
    tom_net = model.PredNet(**model_kwargs)
    tom_net.to(device)

    dicts = dict(past=past, alpha=alpha, batch_size=batch_size,
                 learning_rate=learning_rate, num_epoch=num_epoch, save_freq=save_freq)

    # Make the Dataset
    train_storage = Storage(env, train_population, exp_kwargs['num_past'], exp_kwargs['num_step'])
    eval_storage = Storage(env, eval_population, exp_kwargs['num_past'], exp_kwargs['num_step'])
    train_data = train_storage.extract()
    train_data['exp'] = 'exp1'
    eval_data = eval_storage.extract()
    eval_data['exp'] = 'exp1'
    train_dataset = dataset.ToMDataset(**train_data)
    eval_dataset = dataset.ToMDataset(**eval_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=len(eval_dataset), shuffle=False)

    summary_writer = writer.Writer(experiment_folder)
    visualizer = Visualizer(os.path.join(experiment_folder, 'images'), grid_per_pixel=8,
                            max_epoch=num_epoch, height=env.height, width=env.width)

    # Training loop with metric tracking
    optimizer = optim.Adam(tom_net.parameters(), lr=learning_rate)
    
    train_losses = []
    train_accs = []
    eval_losses = []
    eval_accs = []
    epochs = []

    for epoch in range(dicts['num_epoch']):
        # Train one epoch
        train_results = train_epoch(tom_net, train_loader, optimizer, device=device)
        
        # Evaluate
        eval_results = eval_model(tom_net, eval_loader, device=device, is_visualize=False)

        # Save checkpoint
        if epoch % dicts['save_freq'] == 0:
            utils.save_model(tom_net, dicts, experiment_folder, epoch)
        
        # Log to TensorBoard
        summary_writer.write(train_results, epoch, is_train=True)
        summary_writer.write(eval_results, epoch, is_train=False)
        
        # Print progress
        print('Train| Epoch {} Loss |{:.4f}|Acc |{:.4f}'.format(
            epoch, train_results['action_loss'], train_results['action_acc']))
        print('Eval| Epoch {} Loss |{:.4f}|Acc |{:.4f}'.format(
            epoch, eval_results['action_loss'], eval_results['action_acc']))

        # Track metrics for plotting
        train_losses.append(train_results['action_loss'])
        train_accs.append(train_results['action_acc'])
        eval_losses.append(eval_results['action_loss'])
        eval_accs.append(eval_results['action_acc'])
        epochs.append(epoch)

    # Generate loss curves plot after training
    if visualizer is not None:
        visualizer.plot_loss_curves(train_losses, train_accs, eval_losses, eval_accs, epochs)

    # Save final checkpoint
    final_epoch = dicts['num_epoch'] - 1
    utils.save_model(tom_net, dicts, experiment_folder, final_epoch)

    # Final test evaluation with visualization
    eval_storage.reset()
    test_data = eval_storage.extract()
    test_data['exp'] = 'exp1'
    test_dataset = dataset.ToMDataset(**test_data)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    most_act, count_act = eval_storage.get_most_act()
    
    test_results = eval_model(tom_net, test_loader, device=device, is_visualize=True)
    visualize_results(test_results, visualizer, most_act=most_act, count_act=count_act)
    
    return test_results  # Return final evaluation results



