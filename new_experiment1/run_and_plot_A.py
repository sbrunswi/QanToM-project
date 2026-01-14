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

from experiment1.experiment import run_experiment
from experiment1.config import get_configs
from experiment1.store_trajectories import Storage
from experiment1 import model
from environment.env import GridWorldEnv
from utils import utils
from utils import dataset
from utils.device import get_device
from torch.utils.data import DataLoader


def run_single_experiment(alpha, num_past, num_epoch=100, num_agent=1000, 
                          batch_size=16, lr=1e-4, device='cpu', save_freq=10):
    """Run a single experiment using run_experiment and return final accuracy from model."""
    print(f"\n{'='*60}")
    print(f"Running: alpha={alpha}, N_past={num_past}")
    print(f"{'='*60}")
    
    # Create experiment folder with intuitive naming
    main_experiment = 1  # Experiment 1
    experiment_folder = utils.make_folder(alpha=alpha, num_past=num_past, main_experiment=main_experiment)
    
    try:
        # Run experiment (this trains the model and returns final evaluation results)
        ev_results = run_experiment(
            num_epoch=num_epoch,
            main_experiment=main_experiment,
            sub_experiment=num_past,  # sub_exp controls num_past
            num_agent=num_agent,
            batch_size=batch_size,
            lr=lr,
            experiment_folder=experiment_folder,
            alpha=alpha,
            save_freq=save_freq,
            train_dir='none',
            eval_dir='none',
            device=device
        )
        
        # Get final accuracy directly from the returned evaluation results
        if 'action_acc' not in ev_results:
            print(f"Warning: 'action_acc' not found in evaluation results. Keys: {ev_results.keys()}")
            return None
        
        final_accuracy = ev_results['action_acc']
        
        print(f"Final accuracy: {final_accuracy:.4f}")
        return final_accuracy
        
    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        return None


def _get_epoch_from_filename(filename):
    """Extract epoch number from checkpoint filename (last epoch_XXX, not num_epoch_XXX)."""
    matches = list(re.finditer(r'epoch_(\d+)', filename.name))
    if matches:
        return int(matches[-1].group(1))
    return -1


def _parse_experiment_params_from_folder(folder):
    """Parse alpha, num_past, and main_experiment from folder name or checkpoint names."""
    folder_name = folder.name
    alpha = None
    num_past = None
    main_experiment = None
    
    # Try new format first: _expX_alpha_Y_npast_Z or _alpha_Y_npast_Z
    exp_match = re.search(r'_exp(\d+)', folder_name)
    alpha_match = re.search(r'_alpha_([\d.]+)', folder_name)
    npast_match = re.search(r'_npast_(\d+)', folder_name)
    
    if exp_match:
        main_experiment = int(exp_match.group(1))
    if alpha_match and npast_match:
        alpha = float(alpha_match.group(1))
        num_past = int(npast_match.group(1))
    else:
        # Try to parse from checkpoint names
        checkpoints_dir = folder / 'checkpoints'
        if checkpoints_dir.exists():
            checkpoint_files = list(checkpoints_dir.glob('model_*'))
            if checkpoint_files:
                checkpoint_name = checkpoint_files[0].name
                main_match = re.search(r'main_(\d+)', checkpoint_name)
                alpha_match = re.search(r'alpha_([\d.]+)', checkpoint_name)
                sub_match = re.search(r'sub_(\d+)', checkpoint_name)
                
                if main_match:
                    main_experiment = int(main_match.group(1))
                if alpha_match:
                    alpha = float(alpha_match.group(1))
                if sub_match:
                    num_past = int(sub_match.group(1))
    
    return alpha, num_past, main_experiment


def _load_and_evaluate_checkpoint(checkpoint_path, alpha, num_past, num_agent, device):
    """Load a checkpoint and evaluate it on fresh data."""
    tom_net = tr.load(checkpoint_path, map_location=device, weights_only=False)
    tom_net.eval()
    
    exp_kwargs, env_kwargs, model_kwargs, _ = get_configs(num_past)
    model_kwargs['device'] = device
    
    # Create fresh data for evaluation
    population = utils.make_pool('random', exp_kwargs['move_penalty'], alpha, num_agent)
    env = GridWorldEnv(env_kwargs)
    eval_storage = Storage(env, population, exp_kwargs['num_past'], exp_kwargs['num_step'])
    eval_data = eval_storage.extract()
    eval_data['exp'] = 'exp1'
    eval_dataset = dataset.ToMDataset(**eval_data)
    eval_loader = DataLoader(eval_dataset, batch_size=len(eval_dataset), shuffle=False)
    
    # Evaluate
    with tr.no_grad():
        ev_results = tom_net.evaluate(eval_loader, is_visualize=False)
    
    return ev_results['action_acc']


def parse_existing_results(results_dir=None, num_agent=1000, device='cpu'):
    """
    Parse existing experiment results from the results directory.
    
    Args:
        results_dir: Path to results directory (default: project_root/results)
        num_agent: Number of agents to use for re-evaluation
        device: Device to use for re-evaluation
    
    Returns:
        Dictionary mapping (alpha, num_past) -> accuracy
    """
    if results_dir is None:
        results_dir = project_root / 'results'
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return {}
    
    results = {}
    experiment_folders = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    
    print(f"\nScanning {len(experiment_folders)} experiment folders...")
    
    for folder in experiment_folders:
        # Parse experiment parameters from folder
        alpha, num_past, main_experiment = _parse_experiment_params_from_folder(folder)
        
        if alpha is None or num_past is None:
            continue
        
        # Find the latest checkpoint
        checkpoints_dir = folder / 'checkpoints'
        if not checkpoints_dir.exists() or not checkpoints_dir.is_dir():
            continue
        
        checkpoint_files = list(checkpoints_dir.glob('model_*'))
        if not checkpoint_files:
            continue
        
        # Sort by modification time (most recent last)
        checkpoint_files.sort(key=lambda f: f.stat().st_mtime)
        latest_checkpoint = checkpoint_files[-1]  # Last file by modification time
        
        latest_epoch = _get_epoch_from_filename(latest_checkpoint)
        available_epochs = sorted([_get_epoch_from_filename(f) for f in checkpoint_files], reverse=True)
        
        exp_info = f"exp{main_experiment}, " if main_experiment is not None else ""
        print(f"\nFound experiment: {exp_info}alpha={alpha}, num_past={num_past}")
        print(f"  Folder: {folder.name}")
        print(f"  Available epochs: {available_epochs}")
        print(f"  Latest checkpoint: {latest_checkpoint.name}")
        
        # Skip if only epoch 0 checkpoint exists (untrained model)
        if latest_epoch == 0 and len(available_epochs) == 1:
            print(f"  Skipping: Only epoch 0 checkpoint found (untrained model)")
            continue
        
        # Load model and re-evaluate
        try:
            accuracy = _load_and_evaluate_checkpoint(
                latest_checkpoint, alpha, num_past, num_agent, device
            )
            results[(alpha, num_past)] = accuracy
            print(f"  Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"  Error loading/evaluating: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nParsed {len(results)} results from existing experiments")
    return results



def _find_experiment_folder(alpha, num_past, results_dir=None):
    """Find the folder for an experiment with given alpha and num_past, return folder and latest checkpoint."""
    if results_dir is None:
        results_dir = project_root / 'results'
    
    if not results_dir.exists():
        return None, None
    
    # Look for folders matching the pattern
    pattern = f"*_alpha_{alpha}_npast_{num_past}"
    matching_folders = list(results_dir.glob(pattern))
    
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


def _run_all_experiments(alphas, n_past_values, num_epoch, num_agent, batch_size, lr, device):
    """Run experiments for all combinations of alphas and n_past_values."""
    results = {}
    total_experiments = len(alphas) * len(n_past_values)
    print(f"Total experiments: {len(alphas)} alphas × {len(n_past_values)} N_past = {total_experiments}")
    
    for alpha in alphas:
        for n_past in n_past_values:
            key = (alpha, n_past)
            
            # Check if experiment already exists and load it directly
            folder, checkpoint = _find_experiment_folder(alpha, n_past)
            if checkpoint is not None:
                print(f"\nSkipping: Experiment alpha={alpha}, N_past={n_past} already exists")
                print(f"  Folder: {folder.name}")
                print(f"  Loading checkpoint: {checkpoint.name}")
                # Load the existing result directly
                try:
                    accuracy = _load_and_evaluate_checkpoint(
                        checkpoint, alpha, n_past, num_agent, device
                    )
                    results[key] = accuracy
                    print(f"  Loaded accuracy: {accuracy:.4f}")
                except Exception as e:
                    print(f"  Error loading: {e}")
                    results[key] = None
                continue
            
            accuracy = run_single_experiment(
                alpha=alpha,
                num_past=n_past,
                num_epoch=num_epoch,
                num_agent=num_agent,
                batch_size=batch_size,
                lr=lr,
                device=device
            )
            results[key] = accuracy
    return results


def _find_missing_experiments(results, alphas, n_past_values):
    """Find missing experiment combinations."""
    missing = []
    for alpha in alphas:
        for n_past in n_past_values:
            key = (alpha, n_past)
            if key not in results or results[key] is None:
                missing.append((alpha, n_past))
    return missing


def plot_results(results, save_path=None, alphas=None, n_past_values=None):
    """Create the plot with accuracy vs alpha for different N_past values."""
    # Extract unique values from results if not provided
    if alphas is None:
        alphas = sorted({alpha for alpha, _ in results.keys()})
    if n_past_values is None:
        n_past_values = sorted({n_past for _, n_past in results.keys()})
    
    # Color map for different N_past values
    color_map = {0: 'lightblue', 1: 'mediumblue', 5: 'darkblue', 10: 'navy'}
    default_colors = ['lightblue', 'mediumblue', 'darkblue', 'navy', 'purple', 'red', 'orange']
    
    plt.figure(figsize=(10, 7))
    
    for i, n_past in enumerate(n_past_values):
        accuracies = []
        valid_alphas = []
        
        # Collect data for this N_past
        for alpha in alphas:
            key = (alpha, n_past)
            if key in results and results[key] is not None:
                accuracies.append(results[key])
                valid_alphas.append(alpha)
        
        if not accuracies:
            print(f"Warning: No data for N_past={n_past}")
            continue
        
        accuracies = np.array(accuracies)
        valid_alphas = np.array(valid_alphas)
        
        # Get color for this N_past
        color = color_map.get(n_past, default_colors[i % len(default_colors)])
        
        # Plot data points
        plt.semilogx(valid_alphas, accuracies, marker='o', 
                    color=color, markersize=8, linestyle='None', 
                    label=f'N_past = {n_past}', linewidth=2)
        
    
    plt.xlabel('Trained α', fontsize=12)
    plt.ylabel('prob.', fontsize=12)
    plt.title('(a)', fontsize=14, loc='left')
    plt.xscale('log')
    plt.xlim([0.008, 4])
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend(title='N_past', loc='best', fontsize=10)
    plt.tight_layout()
    
    # Save plot to results directory
    if save_path is None:
        results_dir = project_root / 'results'
        results_dir.mkdir(exist_ok=True)
        save_path = results_dir / 'accuracy_vs_alpha_plot.png'
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()


def main(run_experiments=True, load_from_results=None):
    """
    Main function to run experiments and create plot.
    
    Args:
        run_experiments: If True, run all experiments. If False, only plot.
        load_from_results: Dictionary of pre-computed results {(alpha, n_past): accuracy}
    """
    # Configuration
    alphas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    n_past_values = [0, 1, 5]
    
    # Training parameters
    num_epoch = 100  # Adjust based on your needs
    num_agent = 1000
    batch_size = 16
    lr = 1e-4
    device = get_device('cuda')
    
    results = {}
    
    if load_from_results:
        # Use provided results
        results = load_from_results
        print("Using provided results for plotting...")
    elif run_experiments:
        # Run all experiments
        print("Running experiments for all combinations...")
        print(f"Total experiments: {len(alphas)} alphas × {len(n_past_values)} N_past = {len(alphas) * len(n_past_values)}")
        results = _run_all_experiments(alphas, n_past_values, num_epoch, num_agent, 
                                       batch_size, lr, device)
    else:
        # Try to load from existing results directory
        print("Attempting to load results from existing experiments...")
        parsed_results = parse_existing_results(num_agent=num_agent, device=device)
        
        if parsed_results:
            # Use parsed results, but fill in missing combinations
            results = parsed_results
            print(f"\nLoaded {len(results)} results from existing experiments")
            print("Running missing experiments...")
            
            missing = _find_missing_experiments(results, alphas, n_past_values)
            if missing:
                print(f"Running {len(missing)} missing experiments...")
                missing_results = _run_all_experiments(
                    [a for a, _ in missing], [n for _, n in missing],
                    num_epoch, num_agent, batch_size, lr, device
                )
                results.update(missing_results)
        else:
            print("No existing results found. Running all experiments...")
            results = _run_all_experiments(alphas, n_past_values, num_epoch, num_agent,
                                          batch_size, lr, device)
    
    # Print results summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    # Dynamic header based on n_past_values
    header = f"{'Alpha':<10}"
    for n_past in n_past_values:
        header += f"{'N_past=' + str(n_past):<12}"
    print(header)
    print("-"*60)
    for alpha in alphas:
        row = f"{alpha:<10.2f}"
        for n_past in n_past_values:
            key = (alpha, n_past)
            if key in results and results[key] is not None:
                row += f"{results[key]:<12.4f}"
            else:
                row += f"{'N/A':<12}"
        print(row)
    
    # Create plot
    plot_results(results, alphas=alphas, n_past_values=n_past_values)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run experiments and plot accuracy vs alpha')
    parser.add_argument('--run', action='store_true', 
                       help='Run all experiments (default: False, only plot)')
    parser.add_argument('--num_epoch', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--num_agent', type=int, default=1000,
                       help='Number of agents')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu, cuda, mps)')
    
    args = parser.parse_args()
    
 
    main(run_experiments=args.run)

