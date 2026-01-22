"""
Benchmarking script that runs all 4 configurations:
1. Quantum model with Cross Entropy loss
2. Classical model with Cross Entropy loss
3. Quantum model with KL Divergence loss
4. Classical model with KL Divergence loss

Usage:
    python benchmark.py --alpha 1.0 [other arguments]
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

from new_experiment1.experiment import run_experiment
from utils import utils


def parse_args():
    parser = argparse.ArgumentParser('Benchmarking script for ToM experiments')
    parser.add_argument('--alpha', '-a', type=float, required=True,
                       help='Alpha parameter for agent population')
    parser.add_argument('--num_epoch', '-e', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--past', '-p', type=int, default=1,
                       help='Number of past episodes (num_past)')
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
                       help='Number of qubits for quantum model')
    parser.add_argument('--n_layers', type=int, default=2,
                       help='Number of layers for quantum model')
    parser.add_argument('--skip_completed', action='store_true',
                       help='Skip configurations that already have results')
    
    args = parser.parse_args()
    return args


def create_benchmark_folder(alpha, num_past, model_type, loss_type):
    """
    Create a folder name for benchmark results with model_type and loss_type identifiers.
    
    Args:
        alpha: Alpha parameter for agent population
        num_past: Number of past episodes
        model_type: "quantum" or "classical"
        loss_type: "cross_entropy" or "kl_divergence"
    
    Returns:
        experiment_folder: Path to the experiment folder
    """
    now = datetime.datetime.now(gettz('Asia/Seoul'))
    year, month, day, hour, minutes, sec = str(now.year)[-2:], now.month, now.day, now.hour, now.minute, now.second
    
    # Create folder name with model_type and loss_type
    foldername = '{}_{}_{}_{}_{}_{}_alpha_{}_npast_{}_{}_{}'.format(
        year, month, day, hour, minutes, sec, alpha, num_past, model_type, loss_type)
    
    folder_dir = './results/{}'.format(foldername)
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
    return folder_dir


def print_result_table(description, training_history):
    """Print a formatted table of results for a configuration."""
    print("\n" + "="*80)
    print(f"RESULTS: {description}")
    print("="*80)
    
    if training_history is None:
        print("Results were skipped (already completed).")
        print("="*80 + "\n")
        return
    
    print(f"{'Metric':<20} {'Final Value':<15} {'Min Value':<15} {'Max Value':<15}")
    print("-"*80)
    
    final_train_loss = training_history.get('final_train_loss', 'N/A')
    final_train_acc = training_history.get('final_train_acc', 'N/A')
    final_eval_loss = training_history.get('final_eval_loss', 'N/A')
    final_eval_acc = training_history.get('final_eval_acc', 'N/A')
    
    train_losses = training_history.get('train_losses', [])
    train_accs = training_history.get('train_accs', [])
    eval_losses = training_history.get('eval_losses', [])
    eval_accs = training_history.get('eval_accs', [])
    
    min_train_loss = min(train_losses) if train_losses else 'N/A'
    max_train_loss = max(train_losses) if train_losses else 'N/A'
    min_train_acc = min(train_accs) if train_accs else 'N/A'
    max_train_acc = max(train_accs) if train_accs else 'N/A'
    min_eval_loss = min(eval_losses) if eval_losses else 'N/A'
    max_eval_loss = max(eval_losses) if eval_losses else 'N/A'
    min_eval_acc = min(eval_accs) if eval_accs else 'N/A'
    max_eval_acc = max(eval_accs) if eval_accs else 'N/A'
    
    # Format values for display
    def format_val(val):
        return f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
    
    print(f"{'Train Loss':<20} {format_val(final_train_loss):<15} {format_val(min_train_loss):<15} {format_val(max_train_loss):<15}")
    print(f"{'Train Accuracy':<20} {format_val(final_train_acc):<15} {format_val(min_train_acc):<15} {format_val(max_train_acc):<15}")
    print(f"{'Eval Loss':<20} {format_val(final_eval_loss):<15} {format_val(min_eval_loss):<15} {format_val(max_eval_loss):<15}")
    print(f"{'Eval Accuracy':<20} {format_val(final_eval_acc):<15} {format_val(min_eval_acc):<15} {format_val(max_eval_acc):<15}")
    
    print("="*80 + "\n")


def plot_combined_results(all_results, output_path):
    """Create a combined plot showing loss and accuracy curves for all configurations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define colors and styles for each configuration
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    linestyles = ['-', '--', '-.', ':']
    
    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    for idx, (description, data) in enumerate(all_results.items()):
        if data and 'train_losses' in data and 'epochs' in data:
            epochs = data['epochs']
            train_losses = data['train_losses']
            ax1.plot(epochs, train_losses, label=description, 
                    color=colors[idx % len(colors)], 
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Evaluation Loss
    ax2 = axes[0, 1]
    for idx, (description, data) in enumerate(all_results.items()):
        if data and 'eval_losses' in data and 'epochs' in data:
            epochs = data['epochs']
            eval_losses = data['eval_losses']
            ax2.plot(epochs, eval_losses, label=description,
                    color=colors[idx % len(colors)],
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Evaluation Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training Accuracy
    ax3 = axes[1, 0]
    for idx, (description, data) in enumerate(all_results.items()):
        if data and 'train_accs' in data and 'epochs' in data:
            epochs = data['epochs']
            train_accs = data['train_accs']
            ax3.plot(epochs, train_accs, label=description,
                    color=colors[idx % len(colors)],
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=2, marker='o', markersize=3)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])
    
    # Plot 4: Evaluation Accuracy
    ax4 = axes[1, 1]
    for idx, (description, data) in enumerate(all_results.items()):
        if data and 'eval_accs' in data and 'epochs' in data:
            epochs = data['epochs']
            eval_accs = data['eval_accs']
            ax4.plot(epochs, eval_accs, label=description,
                    color=colors[idx % len(colors)],
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=2, marker='s', markersize=3)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.set_title('Evaluation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10, loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nCombined comparison plot saved to: {output_path}\n")
    plt.close()


def run_benchmark_config(alpha, use_quantum, loss_type, args):
    """
    Run a single benchmark configuration.
    
    Args:
        alpha: Alpha parameter for agent population
        use_quantum: Whether to use quantum model (True) or classical model (False)
        loss_type: Type of loss ("cross_entropy" or "kl_divergence")
        args: Parsed arguments with other configuration parameters
    
    Returns:
        tuple: (experiment_folder, training_history) where training_history is a dict with metrics
    """
    # Determine model type string for folder naming
    model_type = "quantum" if use_quantum else "classical"
    
    # Create descriptive folder name
    experiment_folder = create_benchmark_folder(alpha, args.past, model_type, loss_type)
    
    print("\n" + "="*80)
    print(f"Running: {model_type.upper()} model with {loss_type.upper()} loss")
    print(f"Alpha: {alpha}, Past: {args.past}")
    print(f"Results folder: {experiment_folder}")
    print("="*80 + "\n")
    
    # Check if results already exist (skip if requested)
    if args.skip_completed:
        loss_curve_path = os.path.join(experiment_folder, 'loss_curves.png')
        if os.path.exists(loss_curve_path):
            print(f"Skipping {model_type}-{loss_type}: Results already exist")
            print(f"  Folder: {experiment_folder}\n")
            return experiment_folder, None
    
    # Run the experiment
    try:
        test_results, training_history = run_experiment(
            num_epoch=args.num_epoch,
            past=args.past,
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
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            loss_type=loss_type
        )
        print(f"\n✓ Completed: {model_type.upper()} model with {loss_type.upper()} loss")
        print(f"  Results saved to: {experiment_folder}")
    except Exception as e:
        print(f"\n✗ Error running {model_type}-{loss_type}: {str(e)}\n")
        raise
    
    return experiment_folder, training_history


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("BENCHMARKING SCRIPT")
    print("="*80)
    print(f"Alpha: {args.alpha}")
    print(f"Epochs: {args.num_epoch}")
    print(f"Past episodes: {args.past}")
    print(f"Number of agents: {args.num_agent}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {args.device}")
    print("="*80)
    print("\nWill run 4 configurations:")
    print("  1. Quantum + Cross Entropy")
    print("  2. Classical + Cross Entropy")
    print("  3. Quantum + KL Divergence")
    print("  4. Classical + KL Divergence")
    print("="*80 + "\n")
    
    # Configuration list: (use_quantum, loss_type, description)
    configurations = [
        (True, "cross_entropy", "Quantum + Cross Entropy"),
        (False, "cross_entropy", "Classical + Cross Entropy"),
        (True, "kl_divergence", "Quantum + KL Divergence"),
        (False, "kl_divergence", "Classical + KL Divergence"),
    ]
    
    results = {}  # Stores (folder, training_history) tuples
    all_training_history = {}  # Stores training histories for plotting
    
    for use_quantum, loss_type, description in configurations:
        try:
            experiment_folder, training_history = run_benchmark_config(
                alpha=args.alpha,
                use_quantum=use_quantum,
                loss_type=loss_type,
                args=args
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
        combined_plot_path = f'./results/benchmark_alpha_{args.alpha}_npast_{args.past}_{timestamp}.png'
        
        print("\n" + "="*80)
        print("CREATING COMBINED COMPARISON PLOT")
        print("="*80)
        plot_combined_results(all_training_history, combined_plot_path)
    
    print("All benchmarking completed!")
    
if __name__ == '__main__':
    main()
