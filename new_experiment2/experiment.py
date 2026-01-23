from environment.env import GridWorldEnv
from new_experiment2 import model
from new_experiment2.store_trajectories import Storage
from new_experiment2.config import get_configs
from new_experiment2.trainer import train_epoch, eval_model

from utils.visualize import *
from utils import utils
from utils import dataset
from utils import writer

from torch.utils.data import DataLoader
import torch as tr
import torch.optim as optim
import numpy as np
import os
import glob


def determine_sub_experiment(num_step, move_penalty, height, width):
    """Determine sub-experiment number from parameters."""
    if height == 25 and width == 25:
        return 4
    elif move_penalty == -0.5:
        return 3
    elif num_step == 1:
        return 2
    else:
        return 1


def visualize_results(ev_results, visualizer, preference=None, mode='eval'):
    """Helper function to visualize evaluation results.
    
    Args:
        ev_results: Dictionary from eval_model with visualization data
        visualizer: Visualizer instance
        preference: True preferences for agents (optional)
        mode: Mode string for filename (default: 'eval')
    """
    if 'past_traj' not in ev_results:
        return
    
    past_traj = ev_results.get('past_traj', None)
    curr_state = ev_results.get('curr_state', None)
    pred_actions = ev_results.get('pred_actions', None)
    pred_consumption = ev_results.get('pred_consumption', None)
    pred_sr = ev_results.get('pred_sr', None)
    targ_actions = ev_results.get('targ_actions', None)
    targ_consumption = ev_results.get('targ_consumption', None)
    targ_sr = ev_results.get('targ_sr', None)
    
    if past_traj is None:
        return
    
    indiv_length = len(past_traj)
    env_size = past_traj[0][0][0].shape
    env_height, env_width = env_size[0], env_size[1]
    
    for n in range(indiv_length):
        _, past_actions = np.where(past_traj[n, 0, :, :, :, 6:].sum((1, 2)) == env_height * env_width)
        agent_xys = np.where(past_traj[n, 0, :, :, :, 5] == 1)
        visualizer.get_past_traj(past_traj[n][0][0], agent_xys, past_actions, mode, sample_num=n)
        visualizer.get_curr_state(curr_state[n], mode, sample_num=n)
        
        if pred_actions is not None:
            visualizer.get_action(pred_actions[n], mode, sample_num=n)
        if pred_consumption is not None:
            visualizer.get_prefer(pred_consumption[n], mode, sample_num=n)
        if pred_sr is not None:
            visualizer.get_sr(curr_state[n], pred_sr[n], mode, sample_num=n)
        
        if targ_actions is not None:
            visualizer.get_action(targ_actions[n], mode + '_targ', sample_num=n)
        if targ_consumption is not None:
            visualizer.get_prefer(targ_consumption[n], mode + '_targ', sample_num=n)
        if targ_sr is not None:
            visualizer.get_sr(curr_state[n], targ_sr[n], mode + '_targ', sample_num=n)
    
    # Character embeddings visualization
    if 'e_char' in ev_results:
        e_char = ev_results['e_char']
        num_samples = len(e_char)
        preference_sliced = preference[:num_samples] if preference is not None and len(preference) > num_samples else preference
        visualizer.get_consume_char(e_char, preference_sliced, mode)
        visualizer.tsne_consume_char(e_char, preference_sliced, mode)


def make_dataset(data_dir):
    """Load dataset from saved numpy files."""
    data = {}
    data["episodes"] = np.load(data_dir + "/episodes.npy")
    data["curr_state"] = np.load(data_dir + "/curr_state.npy")
    data["target_action"] = np.load(data_dir + "/target_action.npy")
    data["target_prefer"] = np.load(data_dir + "/target_prefer.npy")
    data["target_sr"] = np.load(data_dir + "/target_sr.npy")
    data['exp'] = 'exp2'
    
    tom_dataset = dataset.ToMDataset(**data)
    return tom_dataset


def run_experiment(num_epoch, num_step, move_penalty, height, width, num_agent, batch_size, learning_rate,
                   experiment_folder, alpha, save_freq, train_dir='none', eval_dir='none', device=None,
                   use_quantum=False, n_qubits=4, n_layers=2, num_past=1):
    """
    Run the Theory of Mind experiment 2.
    
    Args:
        num_epoch: Number of training epochs
        num_step: Number of steps per episode
        move_penalty: Move penalty for agents
        height: Environment height
        width: Environment width
        num_agent: Number of agents in the population
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        experiment_folder: Folder to save experiment results
        alpha: Alpha parameter(s) for agent population
        save_freq: Frequency of checkpoint saving
        train_dir: Directory for training data (default: 'none' - generate on fly)
        eval_dir: Directory for eval data (default: 'none' - generate on fly)
        device: Device to run on ('cpu', 'cuda', 'mps')
        use_quantum: Whether to use quantum-enhanced PredNetQuantum model (default: False)
        n_qubits: Number of qubits for quantum model (default: 4)
        n_layers: Number of layers for quantum model (default: 2)
        num_past: Number of past trajectories/episodes to use (default: 1)
    """
    # Determine sub_experiment from parameters
    sub_experiment = determine_sub_experiment(num_step, move_penalty, height, width)
    
    exp_kwargs, env_kwargs, model_kwargs, agent_type = get_configs(sub_experiment)
    
    # Override config values with provided parameters
    env_kwargs['height'] = height
    env_kwargs['width'] = width
    exp_kwargs['num_step'] = num_step
    exp_kwargs['move_penalty'] = move_penalty
    exp_kwargs['num_past'] = num_past
    model_kwargs['num_step'] = num_step
    model_kwargs['num_past'] = num_past
    
    env = GridWorldEnv(env_kwargs)
    model_kwargs['num_agent'] = num_agent
    model_kwargs['device'] = device
    
    # Choose model based on use_quantum argument
    if use_quantum:
        model_kwargs['n_qubits'] = n_qubits
        model_kwargs['n_layers'] = n_layers
        model_kwargs['use_quantum'] = True
        tom_net = model.PredNetQuantum(**model_kwargs)
    else:
        tom_net = model.PredNet(**model_kwargs)
    
    tom_net.to(device)
    
    dicts = dict(main=2, sub=sub_experiment, alpha=alpha, batch_size=batch_size,
                 lr=learning_rate, num_epoch=num_epoch, save_freq=save_freq)
    
    # Load datasets
    if train_dir != 'none':
        eval_dirs = glob.glob(eval_dir + '*')
        train_dataset = make_dataset(train_dir)
        eval_dataset_1 = make_dataset(eval_dirs[0])
        eval_loader_1 = DataLoader(eval_dataset_1, batch_size=batch_size, shuffle=False)
        eval_loaders = [eval_loader_1]
        
        train_prefer = np.load(train_dir + "/true_prefer.npy")
        test_1_prefer = np.load(eval_dirs[0] + "/true_prefer.npy")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_prefers = [test_1_prefer]
    else:
        if num_agent > 1000:
            num_agent = 1000
        population = utils.make_pool(agent_type, exp_kwargs['move_penalty'], alpha, num_agent)
        # Make the Dataset
        train_storage = Storage(env, population, exp_kwargs['num_past'], exp_kwargs['num_step'])
        eval_storage = Storage(env, population[:num_agent], exp_kwargs['num_past'], exp_kwargs['num_step'])
        train_data = train_storage.extract()
        train_data['exp'] = 'exp2'
        eval_data = eval_storage.extract()
        eval_data['exp'] = 'exp2'
        train_dataset = dataset.ToMDataset(**train_data)
        eval_dataset = dataset.ToMDataset(**eval_data)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=num_agent, shuffle=False)
        eval_loaders = [eval_loader]
        train_prefer = train_storage.true_preference
        eval_prefers = [eval_storage.true_preference]
    
    summary_writer = writer.Writer(experiment_folder)
    visualizer = Visualizer(os.path.join(experiment_folder, 'images'), grid_per_pixel=8,
                            max_epoch=num_epoch, height=env.height, width=env.width)
    
    # Training loop with metric tracking (using trainer functions)
    optimizer = optim.Adam(tom_net.parameters(), lr=learning_rate)
    
    train_total_losses = []
    train_action_losses = []
    train_consumption_losses = []
    train_sr_losses = []
    train_action_accs = []
    train_consumption_accs = []
    eval_total_losses = []
    eval_action_losses = []
    eval_consumption_losses = []
    eval_sr_losses = []
    eval_action_accs = []
    eval_consumption_accs = []
    epochs = []
    
    for epoch in range(num_epoch):
        # Train one epoch using trainer function
        train_results = train_epoch(tom_net, train_loader, optimizer, device=device, num_agent=num_agent)
        
        # Evaluate on first eval loader using trainer function
        eval_results = eval_model(tom_net, eval_loaders[0], device=device, is_visualize=False, num_agent=num_agent)
        
        # Save checkpoint
        if epoch % save_freq == 0:
            utils.save_model(tom_net, dicts, experiment_folder, epoch)
        
        # Log to TensorBoard
        summary_writer.write(train_results, epoch, is_train=True)
        summary_writer.write(eval_results, epoch, is_train=False)
        
        # Print progress
        print('Train| Epoch {} Loss |Total {:.4f} Consume {:.4f} Action {:.4f} SR {:.4f}| Acc |Action {:.4f} Consume {:.4f}|'.format(
            epoch, train_results['total_loss'], train_results['consumption_loss'], 
            train_results['action_loss'], train_results['sr_loss'],
            train_results['action_acc'], train_results['consumption_acc']))
        print('Eval| Epoch {} Loss |Total {:.4f} Consume {:.4f} Action {:.4f} SR {:.4f}| Acc |Action {:.4f} Consume {:.4f}|'.format(
            epoch, eval_results['total_loss'], eval_results['consumption_loss'],
            eval_results['action_loss'], eval_results['sr_loss'],
            eval_results['action_acc'], eval_results['consumption_acc']))
        
        # Track metrics for plotting
        train_total_losses.append(train_results['total_loss'])
        train_action_losses.append(train_results['action_loss'])
        train_consumption_losses.append(train_results['consumption_loss'])
        train_sr_losses.append(train_results['sr_loss'])
        train_action_accs.append(train_results['action_acc'])
        train_consumption_accs.append(train_results['consumption_acc'])
        eval_total_losses.append(eval_results['total_loss'])
        eval_action_losses.append(eval_results['action_loss'])
        eval_consumption_losses.append(eval_results['consumption_loss'])
        eval_sr_losses.append(eval_results['sr_loss'])
        eval_action_accs.append(eval_results['action_acc'])
        eval_consumption_accs.append(eval_results['consumption_acc'])
        epochs.append(epoch)
    
    # Generate loss curves plot after training
    if visualizer is not None:
        # Create simplified loss/accuracy curves (using total loss and action acc for simplicity)
        visualizer.plot_loss_curves(
            train_total_losses, train_action_accs, 
            eval_total_losses, eval_action_accs, 
            epochs
        )
    
    # Save final checkpoint
    final_epoch = num_epoch - 1
    utils.save_model(tom_net, dicts, experiment_folder, final_epoch)
    
    # Final test evaluation with visualization using trainer function
    train_fixed_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    train_results = eval_model(tom_net, train_fixed_loader, device=device, is_visualize=True, num_agent=num_agent)
    visualize_results(train_results, visualizer, preference=train_prefer, mode='train')
    
    # Test on eval sets using trainer function
    for i, (eval_loader, eval_prefer) in enumerate(zip(eval_loaders, eval_prefers)):
        eval_results = eval_model(tom_net, eval_loader, device=device, is_visualize=True, num_agent=num_agent)
        visualize_results(eval_results, visualizer, preference=eval_prefer, mode='eval{}'.format(i))
    
    # Return final evaluation results and training history
    training_history = {
        'train_total_losses': train_total_losses,
        'train_action_losses': train_action_losses,
        'train_consumption_losses': train_consumption_losses,
        'train_sr_losses': train_sr_losses,
        'train_action_accs': train_action_accs,
        'train_consumption_accs': train_consumption_accs,
        'eval_total_losses': eval_total_losses,
        'eval_action_losses': eval_action_losses,
        'eval_consumption_losses': eval_consumption_losses,
        'eval_sr_losses': eval_sr_losses,
        'eval_action_accs': eval_action_accs,
        'eval_consumption_accs': eval_consumption_accs,
        'epochs': epochs,
        'final_train_total_loss': train_total_losses[-1] if train_total_losses else None,
        'final_train_action_acc': train_action_accs[-1] if train_action_accs else None,
        'final_eval_total_loss': eval_total_losses[-1] if eval_total_losses else None,
        'final_eval_action_acc': eval_action_accs[-1] if eval_action_accs else None,
    }
    
    return eval_results, training_history
