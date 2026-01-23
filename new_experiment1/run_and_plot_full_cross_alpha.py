import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re
import torch as tr
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from new_experiment1.experiment import run_experiment
from new_experiment1.config import get_configs
from new_experiment1.store_trajectories import Storage
from new_experiment1 import model
from environment.env import GridWorldEnv
from utils import utils
from utils import dataset
from utils.device import get_device
from torch.utils.data import DataLoader


def normalize_alpha_key(alpha):
    """Helper function to normalize alpha keys for dictionary lookup."""
    if isinstance(alpha, (list, np.ndarray)):
        return tuple(sorted(alpha) if isinstance(alpha, list) else sorted(alpha.tolist()))
    return alpha


def train_model_on_alpha(train_alpha, num_past, num_epoch=100, num_agent=1000, 
                         batch_size=16, lr=1e-4, device='cpu', save_freq=10,
                         use_quantum=False, n_qubits=4, n_layers=2):
    """Train a model on a specific alpha (or list of alphas) and return the trained model path."""
    model_type = "quantum" if use_quantum else "classical"
    print(f"\n{'='*60}")
    print(f"Training {model_type} model on alpha={train_alpha}, N_past={num_past}")
    if use_quantum:
        print(f"Quantum parameters: n_qubits={n_qubits}, n_layers={n_layers}")
    print(f"{'='*60}")
    
    # Create experiment folder with intuitive naming
    if isinstance(train_alpha, (list, np.ndarray)):
        alpha_str = '_'.join([str(a) for a in train_alpha])
    else:
        alpha_str = str(train_alpha)
    
    experiment_folder = utils.make_folder(alpha=alpha_str, num_past=num_past, main_experiment=1)
    
    try:
        # Convert train_alpha to numpy array if it's a list
        if isinstance(train_alpha, list):
            train_alpha = np.array(train_alpha)
        
        # Run experiment (this trains the model)
        run_experiment(
            num_epoch=num_epoch,
            past=num_past,
            num_agent=num_agent,
            batch_size=batch_size,
            learning_rate=lr,
            experiment_folder=experiment_folder,
            alpha=train_alpha,
            save_freq=save_freq,
            train_dir='none',
            eval_dir='none',
            device=device,
            use_quantum=use_quantum,
            n_qubits=n_qubits,
            n_layers=n_layers
        )
        
        # Find the latest checkpoint
        checkpoints_dir = Path(experiment_folder) / 'checkpoints'
        checkpoint_files = list(checkpoints_dir.glob('model_*'))
        if not checkpoint_files:
            print(f"Warning: No checkpoint found in {checkpoints_dir}")
            return None
        
        # Sort by modification time and get the latest
        checkpoint_files.sort(key=lambda f: f.stat().st_mtime)
        latest_checkpoint = checkpoint_files[-1]
        
        print(f"Training complete. Latest checkpoint: {latest_checkpoint.name}")
        return latest_checkpoint
        
    except Exception as e:
        print(f"Error training model: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_model_on_test_alpha(checkpoint_path, test_alpha, num_past, 
                                  num_agent=1000, device='cpu'):
    """Load a trained model and evaluate it on a test alpha."""
    try:
        # Ensure device is a torch.device object
        if isinstance(device, str):
            device = tr.device(device)
        
        # Load the trained model
        tom_net = tr.load(checkpoint_path, map_location=device, weights_only=False)
        tom_net.to(device)  # Ensure model is on the correct device
        # Update the device attribute in the model (used for creating new tensors)
        tom_net.device = device
        if hasattr(tom_net, 'e_char'):
            tom_net.e_char.device = device
        tom_net.eval()
        
        # Get configs
        exp_kwargs, env_kwargs, model_kwargs, _ = get_configs(num_past)
        model_kwargs['device'] = device
        
        # Create test population with test_alpha
        test_population = utils.make_pool('random', exp_kwargs['move_penalty'], test_alpha, num_agent)
        env = GridWorldEnv(env_kwargs)
        
        # Create test data
        test_storage = Storage(env, test_population, exp_kwargs['num_past'], exp_kwargs['num_step'])
        test_data = test_storage.extract()
        test_data['exp'] = 'exp1'
        test_dataset = dataset.ToMDataset(**test_data)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        
        # Evaluate
        from new_experiment1.trainer import eval_model
        with tr.no_grad():
            ev_results = eval_model(tom_net, test_loader, device=device, is_visualize=False)
        
        # Return KL divergence (action_loss is average KL divergence)
        kl_div = ev_results['action_loss']
        return kl_div
        
    except Exception as e:
        print(f"Error evaluating model: {e}")
        import traceback
        traceback.print_exc()
        return None


def _get_epoch_from_filename(filename):
    """Extract epoch number from checkpoint filename (last epoch_XXX)."""
    matches = list(re.finditer(r'epoch_(\d+)', filename.name))
    if matches:
        return int(matches[-1].group(1))
    return -1


def _find_trained_model_folder(train_alpha, num_past, results_dir=None, use_quantum=False):
    """Find the folder for a trained model with given train_alpha and num_past."""
    if results_dir is None:
        # Look in new_experiment1/results by default
        results_dir = Path(__file__).parent / 'results'
    
    if not results_dir.exists():
        return None, None
    
    # Create search pattern based on train_alpha
    if isinstance(train_alpha, (list, np.ndarray)):
        alpha_str = '_'.join([str(a) for a in train_alpha])
    else:
        alpha_str = str(train_alpha)
    
    pattern = f"*_alpha_{alpha_str}_npast_{num_past}"
    matching_folders = list(results_dir.glob(pattern))
    
    # Filter by quantum model if needed (check checkpoint file names or folder structure)
    if use_quantum:
        # For quantum models, we might need to check the checkpoint or model type
        # For now, we'll check all folders and let the loading handle it
        pass
    
    # Find the folder with the most recent valid checkpoint
    best_folder = None
    best_checkpoint = None
    best_epoch = -1
    
    for folder in matching_folders:
        checkpoints_dir = folder / 'checkpoints'
        if checkpoints_dir.exists():
            checkpoint_files = list(checkpoints_dir.glob('model_*'))
            if checkpoint_files:
                checkpoint_files.sort(key=lambda f: f.stat().st_mtime)
                latest_checkpoint = checkpoint_files[-1]
                latest_epoch = _get_epoch_from_filename(latest_checkpoint)
                if latest_epoch > best_epoch:
                    best_epoch = latest_epoch
                    best_folder = folder
                    best_checkpoint = latest_checkpoint
    
    if best_epoch > 0:
        return best_folder, best_checkpoint
    return None, None


def run_cross_alpha_experiments(train_alphas, test_alphas, num_past=1, num_epoch=100, 
                                 num_agent=1000, batch_size=16, lr=1e-4, device='cpu', 
                                 save_freq=10, use_quantum=False, n_qubits=4, n_layers=2):
    """
    Run cross-alpha experiments: train on train_alphas, test on test_alphas.
    Uses existing models if available, otherwise trains new ones.
    
    Args:
        train_alphas: List of alpha values (or lists/arrays for mixed) to train on
        test_alphas: List of alpha values to test on
        num_past: Number of past episodes
        num_epoch: Number of training epochs
        num_agent: Number of agents
        batch_size: Batch size
        lr: Learning rate
        device: Device to use
        save_freq: Checkpoint save frequency
        use_quantum: Whether to use quantum-enhanced model
        n_qubits: Number of qubits for quantum model
        n_layers: Number of layers for quantum model
    
    Returns:
        Dictionary mapping (train_alpha, test_alpha) -> kl_divergence
    """
    results = {}
    
    for train_alpha in train_alphas:
        train_key = normalize_alpha_key(train_alpha)
        
        # Check if model already exists
        folder, checkpoint = _find_trained_model_folder(train_alpha, num_past, use_quantum=use_quantum)
        
        if checkpoint is None:
            # No existing model found, train a new one
            model_type = "quantum" if use_quantum else "classical"
            print(f"\nNo existing {model_type} model found for train_alpha={train_alpha}, training new model...")
            checkpoint = train_model_on_alpha(
                train_alpha=train_alpha,
                num_past=num_past,
                num_epoch=num_epoch,
                num_agent=num_agent,
                batch_size=batch_size,
                lr=lr,
                device=device,
                save_freq=save_freq,
                use_quantum=use_quantum,
                n_qubits=n_qubits,
                n_layers=n_layers
            )
            
            if checkpoint is None:
                print(f"Failed to train model for train_alpha={train_alpha}")
                continue
        else:
            # Use existing model
            print(f"\nFound existing model for train_alpha={train_alpha}")
            print(f"  Folder: {folder.name}")
            print(f"  Checkpoint: {checkpoint.name}")
        
        # Test on all test alphas
        print(f"\nTesting model trained on alpha={train_alpha} on test alphas...")
        for test_alpha in test_alphas:
            print(f"  Testing on alpha={test_alpha}...", end=' ', flush=True)
            
            kl_div = evaluate_model_on_test_alpha(
                checkpoint_path=checkpoint,
                test_alpha=test_alpha,
                num_past=num_past,
                num_agent=num_agent,
                device=device
            )
            
            if kl_div is not None:
                results[(train_key, test_alpha)] = kl_div
                print(f"KL-divergence: {kl_div:.4f}")
            else:
                print("Failed")
    
    return results


def plot_cross_alpha_results(results, train_alphas, test_alphas, save_path=None, use_quantum=False):
    """Create the plot showing KL-divergence vs test alpha for each trained alpha."""
    # Color map for different trained alphas (similar to run_and_plot_A.py style)
    # Using blue color progression: darker for smaller alphas, lighter for larger alphas
    style_map = {
        0.01: {'color': 'navy', 'marker': 'o', 'label': '0.01'},  # Darkest blue
        0.03: {'color': 'darkblue', 'marker': 'o', 'label': '0.03'},  # Dark blue
        0.1: {'color': 'mediumblue', 'marker': 'o', 'label': '0.1'},  # Medium blue
        0.3: {'color': 'blue', 'marker': 'o', 'label': '0.3'},  # Standard blue
        1: {'color': 'cornflowerblue', 'marker': 'o', 'label': '1'},  # Medium-light blue
        3: {'color': 'lightblue', 'marker': 'o', 'label': '3'},  # Lightest blue
    }
    # Default colors fallback (similar to run_and_plot_A.py)
    default_colors = ['lightblue', 'mediumblue', 'darkblue', 'navy', 'purple', 'red', 'orange']
    
    plt.figure(figsize=(10, 7))
    
    for train_alpha in train_alphas:
        train_key = normalize_alpha_key(train_alpha)
        
        # Get style for this train_alpha
        if train_key in style_map:
            style = style_map[train_key]
        else:
            # Default style - use default colors similar to run_and_plot_A.py
            # Find index in train_alphas list for consistent color assignment
            try:
                idx = train_alphas.index(train_alpha) if train_alpha in train_alphas else len(train_alphas)
                default_color = default_colors[idx % len(default_colors)]
            except (ValueError, TypeError):
                default_color = 'gray'
            style = {'color': default_color, 'marker': 'o', 'label': str(train_alpha)}
        
        # Collect KL-divergences for all test alphas
        kl_divs = []
        valid_test_alphas = []
        
        for test_alpha in test_alphas:
            key = (train_key, test_alpha)
            if key in results and results[key] is not None:
                kl_divs.append(results[key])
                valid_test_alphas.append(test_alpha)
        
        if not kl_divs:
            print(f"Warning: No data for train_alpha={train_alpha}")
            continue
        
        kl_divs = np.array(kl_divs)
        valid_test_alphas = np.array(valid_test_alphas)
        
        # Plot data points
        plt.semilogx(valid_test_alphas, kl_divs, 
                    marker=style['marker'], 
                    color=style['color'], 
                    markersize=8, 
                    linestyle='-', 
                    linewidth=2,
                    label=f"Trained α = {style['label']}")
    
    plt.xlabel('Test α', fontsize=12)
    plt.ylabel('Average KL-divergence', fontsize=12)
    title = '(a) Quantum Model' if use_quantum else '(a)'
    plt.title(title, fontsize=14, loc='left')
    plt.xscale('log')
    plt.xlim([0.008, 4])
    plt.ylim(bottom=0)  # Let y-axis auto-scale based on data
    plt.grid(True, alpha=0.3)
    plt.legend(title='Trained α', loc='best', fontsize=10)
    plt.tight_layout()
    
    # Save plot to results directory
    if save_path is None:
        results_dir = project_root / 'results'
        results_dir.mkdir(exist_ok=True)
        model_suffix = '_quantum' if use_quantum else ''
        save_path = results_dir / f'full_cross_alpha_plot{model_suffix}.png'
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()


def main(num_epoch=100, num_agent=1000, batch_size=16, device=None, 
         use_quantum=False, n_qubits=4, n_layers=2):
    """
    Main function to run cross-alpha experiments and create plot.
    Uses existing models if available, otherwise trains new ones.
    
    Args:
        num_epoch: Number of training epochs
        num_agent: Number of agents
        batch_size: Batch size
        device: Device to use (if None, will auto-detect)
        use_quantum: Whether to use quantum-enhanced model
        n_qubits: Number of qubits for quantum model
        n_layers: Number of layers for quantum model
    """
    # Configuration
    train_alphas = [0.01, 0.03, 0.1, 0.3, 1, 3]  # Train on these alphas
    test_alphas = [0.01, 0.03, 0.1, 0.3, 1, 3]  # Test on these alphas
    
    # Training parameters
    num_past = 1   # Number of past episodes
    lr = 1e-4
    if device is None:
        device = get_device('cuda')
    save_freq = 10
    
    model_type = "quantum" if use_quantum else "classical"
    print("Running cross-alpha experiments...")
    print(f"Model type: {model_type}")
    if use_quantum:
        print(f"Quantum parameters: n_qubits={n_qubits}, n_layers={n_layers}")
    print(f"Training on alphas: {train_alphas}")
    print(f"Testing on alphas: {test_alphas}")
    print(f"Total experiments: {len(train_alphas)} trained models × {len(test_alphas)} test alphas = {len(train_alphas) * len(test_alphas)}")
    
    results = run_cross_alpha_experiments(
        train_alphas=train_alphas,
        test_alphas=test_alphas,
        num_past=num_past,
        num_epoch=num_epoch,
        num_agent=num_agent,
        batch_size=batch_size,
        lr=lr,
        device=device,
        save_freq=save_freq,
        use_quantum=use_quantum,
        n_qubits=n_qubits,
        n_layers=n_layers
    )
    
    # Print results summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Create header
    header = f"{'Train α':<15}"
    for test_alpha in test_alphas:
        header += f"{'Test=' + str(test_alpha):<12}"
    print(header)
    print("-"*60)
    
    # Print rows for each train_alpha
    for train_alpha in train_alphas:
        if isinstance(train_alpha, (list, np.ndarray)):
            train_str = ', '.join([str(a) for a in train_alpha])
        else:
            train_str = str(train_alpha)
        
        row = f"{train_str:<15}"
        train_key = normalize_alpha_key(train_alpha) if isinstance(train_alpha, (list, np.ndarray)) else train_alpha
        
        for test_alpha in test_alphas:
            key = (train_key, test_alpha)
            if key in results and results[key] is not None:
                row += f"{results[key]:<12.4f}"
            else:
                row += f"{'N/A':<12}"
        print(row)
    
    # Create plot
    plot_cross_alpha_results(results, train_alphas, test_alphas, use_quantum=use_quantum)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run cross-alpha experiments and plot results')
    parser.add_argument('--num_epoch', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--num_agent', type=int, default=1000,
                       help='Number of agents')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu, cuda, mps)')
    parser.add_argument('--use_quantum', action='store_true',
                       help='Use quantum-enhanced PredNetQuantum model')
    parser.add_argument('--n_qubits', type=int, default=3,
                       help='Number of qubits for quantum model')
    parser.add_argument('--n_layers', type=int, default=2,
                       help='Number of layers for quantum model')
    
    args = parser.parse_args()
    
    # Convert device string to device object
    device = get_device(args.device) if args.device != 'cpu' else 'cpu'
    
    main(num_epoch=args.num_epoch, num_agent=args.num_agent, 
         batch_size=args.batch_size, device=device,
         use_quantum=args.use_quantum, n_qubits=args.n_qubits, n_layers=args.n_layers)
