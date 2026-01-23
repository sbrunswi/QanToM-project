"""
Benchmarking script for Experiment 2 that runs multiple model configurations.

Default mode (4 configurations):
1. Quantum model with default settings
2. Classical model with default settings
3. Quantum model with different sub-experiment
4. Classical model with different sub-experiment

With --qubit_sweep (8 configurations by default):
- Quantum models with 3, 4, 5 qubits = 3 quantum configs (per sub-experiment)
- Classical model = 1 classical config (per sub-experiment)

Usage:
    python benchmark.py --alpha 1.0 [other arguments]
    python benchmark.py --alpha 1.0 --qubit_sweep  # Test 3,4,5 qubits
    python benchmark.py --alpha 1.0 --qubit_sweep --qubit_values 2 3 4 5 6  # Custom qubit range
"""

import argparse
import sys
import os
import datetime
from dateutil.tz import gettz
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from new_experiment2.experiment import run_experiment
from utils import utils


def parse_args():
    parser = argparse.ArgumentParser('Benchmarking script for ToM Experiment 2')
    parser.add_argument('--alpha', '-a', type=float, required=True,
                       help='Alpha parameter for agent population')
    parser.add_argument('--num_epoch', '-e', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--past', '-p', type=int, default=1,
                       help='Number of past episodes (num_past)')
    parser.add_argument('--num_step', type=int, default=31,
                       help='Number of steps per episode')
    parser.add_argument('--move_penalty', type=float, default=-0.01,
                       help='Move penalty for agents')
    parser.add_argument('--height', type=int, default=11,
                       help='Environment height')
    parser.add_argument('--width', type=int, default=11,
                       help='Environment width')
    parser.add_argument('--num_agent', '-na', type=int, default=1000,
                       help='Number of agents in the population')
    parser.add_argument('--batch_size', '-b', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', '--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--save_freq', '-s', type=int, default=10,
                       help='Frequency of checkpoint saving')
    parser.add_argument('--train_dir', default='none', type=str,
                       help='Directory for training data')
    parser.add_argument('--eval_dir', default='none', type=str,
                       help='Directory for eval data')
    parser.add_argument('--device', default='cpu',
                       help="Device to use: 'cuda', 'mps', or 'cpu'")
    parser.add_argument('--n_qubits', type=int, default=4,
                       help='Number of qubits for quantum model (ignored if --qubit_sweep is used)')
    parser.add_argument('--n_layers', type=int, default=2,
                       help='Number of layers for quantum model')
    parser.add_argument('--skip_completed', action='store_true',
                       help='Skip configurations that already have results')
    parser.add_argument('--qubit_sweep', action='store_true',
                       help='Run quantum models with 3, 4, and 5 qubits each')
    parser.add_argument('--qubit_values', type=int, nargs='+', default=[3, 4, 5],
                       help='Qubit values to test when --qubit_sweep is enabled (default: 3 4 5)')
    
    args = parser.parse_args()
    return args


def create_benchmark_folder(alpha, num_past, model_type, num_step, move_penalty, height, width, n_qubits=None):
    """
    Create a folder name for benchmark results with model_type and environment identifiers.
    
    Args:
        alpha: Alpha parameter for agent population
        num_past: Number of past episodes
        model_type: "quantum" or "classical"
        num_step: Number of steps per episode
        move_penalty: Move penalty for agents
        height: Environment height
        width: Environment width
        n_qubits: Number of qubits (only for quantum models)
    
    Returns:
        experiment_folder: Path to the experiment folder
    """
    now = datetime.datetime.now(gettz('Asia/Seoul'))
    year, month, day, hour, minutes, sec = str(now.year)[-2:], now.month, now.day, now.hour, now.minute, now.second
    
    # Create folder name with model_type and environment info
    env_info = f"step_{num_step}_h{height}w{width}"
    if model_type == "quantum" and n_qubits is not None:
        foldername = '{}_{}_{}_{}_{}_{}_exp2_alpha_{}_npast_{}_{}_qubits_{}_{}'.format(
            year, month, day, hour, minutes, sec, alpha, num_past, model_type, n_qubits, env_info)
    else:
        foldername = '{}_{}_{}_{}_{}_{}_exp2_alpha_{}_npast_{}_{}_{}'.format(
            year, month, day, hour, minutes, sec, alpha, num_past, model_type, env_info)
    
    folder_dir = './results/{}'.format(foldername)
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
    return folder_dir


def print_result_table(description, training_history):
    """Print a formatted table of results for a configuration."""
    print("\n" + "="*100)
    print(f"RESULTS: {description}")
    print("="*100)
    
    if training_history is None:
        print("Results were skipped (already completed).")
        print("="*100 + "\n")
        return
    
    print(f"{'Metric':<25} {'Final Value':<15} {'Min Value':<15} {'Max Value':<15}")
    print("-"*100)
    
    # Format values for display
    def format_val(val):
        return f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
    
    # Total loss
    total_losses = training_history.get('eval_total_losses', [])
    final_total = training_history.get('final_eval_total_loss', 'N/A')
    min_total = min(total_losses) if total_losses else 'N/A'
    max_total = max(total_losses) if total_losses else 'N/A'
    print(f"{'Total Loss':<25} {format_val(final_total):<15} {format_val(min_total):<15} {format_val(max_total):<15}")
    
    # Action loss
    action_losses = training_history.get('eval_action_losses', [])
    final_action_loss = action_losses[-1] if action_losses else 'N/A'
    min_action = min(action_losses) if action_losses else 'N/A'
    max_action = max(action_losses) if action_losses else 'N/A'
    print(f"{'Action Loss':<25} {format_val(final_action_loss):<15} {format_val(min_action):<15} {format_val(max_action):<15}")
    
    # Consumption loss
    consumption_losses = training_history.get('eval_consumption_losses', [])
    final_cons_loss = consumption_losses[-1] if consumption_losses else 'N/A'
    min_cons = min(consumption_losses) if consumption_losses else 'N/A'
    max_cons = max(consumption_losses) if consumption_losses else 'N/A'
    print(f"{'Consumption Loss':<25} {format_val(final_cons_loss):<15} {format_val(min_cons):<15} {format_val(max_cons):<15}")
    
    # SR loss
    sr_losses = training_history.get('eval_sr_losses', [])
    final_sr_loss = sr_losses[-1] if sr_losses else 'N/A'
    min_sr = min(sr_losses) if sr_losses else 'N/A'
    max_sr = max(sr_losses) if sr_losses else 'N/A'
    print(f"{'SR Loss':<25} {format_val(final_sr_loss):<15} {format_val(min_sr):<15} {format_val(max_sr):<15}")
    
    # Action accuracy
    action_accs = training_history.get('eval_action_accs', [])
    final_action_acc = training_history.get('final_eval_action_acc', 'N/A')
    min_action_acc = min(action_accs) if action_accs else 'N/A'
    max_action_acc = max(action_accs) if action_accs else 'N/A'
    print(f"{'Action Accuracy':<25} {format_val(final_action_acc):<15} {format_val(min_action_acc):<15} {format_val(max_action_acc):<15}")
    
    # Consumption accuracy
    consumption_accs = training_history.get('eval_consumption_accs', [])
    final_cons_acc = consumption_accs[-1] if consumption_accs else 'N/A'
    min_cons_acc = min(consumption_accs) if consumption_accs else 'N/A'
    max_cons_acc = max(consumption_accs) if consumption_accs else 'N/A'
    print(f"{'Consumption Accuracy':<25} {format_val(final_cons_acc):<15} {format_val(min_cons_acc):<15} {format_val(max_cons_acc):<15}")
    
    print("="*100 + "\n")


def plot_combined_results(all_results, output_path):
    """Create a combined plot showing loss and accuracy curves for all configurations."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # Define colors and styles for each configuration
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    
    # Plot 1: Total Loss (Train)
    ax1 = axes[0, 0]
    for idx, (description, data) in enumerate(all_results.items()):
        if data and 'train_total_losses' in data and 'epochs' in data:
            epochs = data['epochs']
            losses = data['train_total_losses']
            ax1.plot(epochs, losses, label=description, 
                    color=colors[idx % len(colors)], 
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=2, marker='o', markersize=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Total Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Total Loss (Eval)
    ax2 = axes[0, 1]
    for idx, (description, data) in enumerate(all_results.items()):
        if data and 'eval_total_losses' in data and 'epochs' in data:
            epochs = data['epochs']
            losses = data['eval_total_losses']
            ax2.plot(epochs, losses, label=description,
                    color=colors[idx % len(colors)],
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=2, marker='s', markersize=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Evaluation Total Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Action Accuracy (Train)
    ax3 = axes[1, 0]
    for idx, (description, data) in enumerate(all_results.items()):
        if data and 'train_action_accs' in data and 'epochs' in data:
            epochs = data['epochs']
            accs = data['train_action_accs']
            ax3.plot(epochs, accs, label=description,
                    color=colors[idx % len(colors)],
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=2, marker='o', markersize=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Training Action Accuracy', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=8, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])
    
    # Plot 4: Action Accuracy (Eval)
    ax4 = axes[1, 1]
    for idx, (description, data) in enumerate(all_results.items()):
        if data and 'eval_action_accs' in data and 'epochs' in data:
            epochs = data['epochs']
            accs = data['eval_action_accs']
            ax4.plot(epochs, accs, label=description,
                    color=colors[idx % len(colors)],
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=2, marker='s', markersize=2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.set_title('Evaluation Action Accuracy', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=8, loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.05])
    
    # Plot 5: Consumption Accuracy (Train)
    ax5 = axes[2, 0]
    for idx, (description, data) in enumerate(all_results.items()):
        if data and 'train_consumption_accs' in data and 'epochs' in data:
            epochs = data['epochs']
            accs = data['train_consumption_accs']
            ax5.plot(epochs, accs, label=description,
                    color=colors[idx % len(colors)],
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=2, marker='o', markersize=2)
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Accuracy', fontsize=12)
    ax5.set_title('Training Consumption Accuracy', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=8, loc='best')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1.05])
    
    # Plot 6: Consumption Accuracy (Eval)
    ax6 = axes[2, 1]
    for idx, (description, data) in enumerate(all_results.items()):
        if data and 'eval_consumption_accs' in data and 'epochs' in data:
            epochs = data['epochs']
            accs = data['eval_consumption_accs']
            ax6.plot(epochs, accs, label=description,
                    color=colors[idx % len(colors)],
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=2, marker='s', markersize=2)
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('Accuracy', fontsize=12)
    ax6.set_title('Evaluation Consumption Accuracy', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=8, loc='best')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nCombined comparison plot saved to: {output_path}\n")
    plt.close()


def run_benchmark_config(alpha, use_quantum, args, n_qubits=None):
    """
    Run a single benchmark configuration.
    
    Args:
        alpha: Alpha parameter for agent population
        use_quantum: Whether to use quantum model (True) or classical model (False)
        args: Parsed arguments with other configuration parameters
        n_qubits: Number of qubits for quantum model (overrides args.n_qubits if provided)
    
    Returns:
        tuple: (experiment_folder, training_history) where training_history is a dict with metrics
    """
    # Determine model type string for folder naming
    model_type = "quantum" if use_quantum else "classical"
    
    # Use provided n_qubits or fall back to args
    actual_n_qubits = n_qubits if n_qubits is not None else args.n_qubits
    
    # Create descriptive folder name
    experiment_folder = create_benchmark_folder(
        alpha, args.past, model_type, 
        args.num_step, args.move_penalty, args.height, args.width,
        n_qubits=actual_n_qubits if use_quantum else None
    )
    
    qubit_info = f", Qubits: {actual_n_qubits}" if use_quantum else ""
    print("\n" + "="*80)
    print(f"Running: {model_type.upper()} model")
    print(f"Alpha: {alpha}, Past: {args.past}{qubit_info}")
    print(f"Environment: {args.height}x{args.width}, Steps: {args.num_step}, Move penalty: {args.move_penalty}")
    print(f"Results folder: {experiment_folder}")
    print("="*80 + "\n")
    
    # Check if results already exist (skip if requested)
    if args.skip_completed:
        loss_curve_path = os.path.join(experiment_folder, 'loss_curves.png')
        if os.path.exists(loss_curve_path):
            print(f"Skipping {model_type}: Results already exist")
            print(f"  Folder: {experiment_folder}\n")
            return experiment_folder, None
    
    # Run the experiment
    try:
        test_results, training_history = run_experiment(
            num_epoch=args.num_epoch,
            num_step=args.num_step,
            move_penalty=args.move_penalty,
            height=args.height,
            width=args.width,
            num_agent=args.num_agent,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            experiment_folder=experiment_folder,
            alpha=alpha,
            save_freq=args.save_freq,
            train_dir=args.train_dir,
            eval_dir=args.eval_dir,
            device=args.device,
            use_quantum=use_quantum,
            n_qubits=actual_n_qubits,
            n_layers=args.n_layers,
            num_past=args.past
        )
        print(f"\n✓ Completed: {model_type.upper()} model{qubit_info}")
        print(f"  Results saved to: {experiment_folder}")
    except Exception as e:
        print(f"\n✗ Error running {model_type}: {str(e)}\n")
        raise
    
    return experiment_folder, training_history


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("BENCHMARKING SCRIPT - EXPERIMENT 2")
    print("="*80)
    print(f"Alpha: {args.alpha}")
    print(f"Epochs: {args.num_epoch}")
    print(f"Past episodes: {args.past}")
    print(f"Environment: {args.height}x{args.width}")
    print(f"Steps per episode: {args.num_step}")
    print(f"Move penalty: {args.move_penalty}")
    print(f"Number of agents: {args.num_agent}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {args.device}")
    if args.qubit_sweep:
        print(f"Qubit sweep enabled: {args.qubit_values}")
    else:
        print(f"Qubits (for quantum): {args.n_qubits}")
    print("="*80)
    
    # Build configuration list based on qubit_sweep flag
    configurations = []
    
    if args.qubit_sweep:
        # With qubit sweep: test each qubit count for quantum models
        for n_qubits in args.qubit_values:
            configurations.append((True, n_qubits, f"Quantum ({n_qubits}q)"))
        # Add classical model (only once, no qubit variation)
        configurations.append((False, None, "Classical"))
        
        print(f"\nWill run {len(configurations)} configurations:")
        for i, (use_q, n_q, desc) in enumerate(configurations, 1):
            print(f"  {i}. {desc}")
    else:
        # Without qubit sweep: just quantum vs classical
        configurations = [
            (True, args.n_qubits, "Quantum"),
            (False, None, "Classical"),
        ]
        print("\nWill run 2 configurations:")
        print("  1. Quantum")
        print("  2. Classical")
    
    print("="*80 + "\n")
    
    results = {}  # Stores (folder, training_history) tuples
    all_training_history = {}  # Stores training histories for plotting
    
    for use_quantum, n_qubits, description in configurations:
        try:
            experiment_folder, training_history = run_benchmark_config(
                alpha=args.alpha,
                use_quantum=use_quantum,
                args=args,
                n_qubits=n_qubits
            )
            results[description] = experiment_folder
            all_training_history[description] = training_history
            
            # Print result table after each configuration
            if training_history:
                print_result_table(description, training_history)
                
        except KeyboardInterrupt:
            print("\n\nBenchmarking interrupted by user.")
            print(f"Completed configurations: {list(results.keys())}")
            sys.exit(1)
        except Exception as e:
            print(f"\nFailed to run {description}: {str(e)}")
            print("Continuing with next configuration...\n")
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARKING SUMMARY")
    print("="*80)
    for description, folder in results.items():
        print(f"✓ {description}")
        print(f"  {folder}")
    print("="*80 + "\n")
    
    # Create combined comparison plot
    if all_training_history and any(h is not None for h in all_training_history.values()):
        # Determine output path for combined plot (save in results directory with timestamp)
        timestamp = datetime.datetime.now(gettz('Asia/Seoul')).strftime('%y_%m_%d_%H_%M_%S')
        combined_plot_path = f'./results/benchmark_exp2_alpha_{args.alpha}_npast_{args.past}_{timestamp}.png'
        
        print("\n" + "="*80)
        print("CREATING COMBINED COMPARISON PLOT")
        print("="*80)
        plot_combined_results(all_training_history, combined_plot_path)
    
    print("All benchmarking completed!")

    
if __name__ == '__main__':
    main()
