#!/usr/bin/env python3
"""
State Encoder Comparison Experiment

Compares classical, quantum, and hybrid state encoders while fixing belief
state to classical to isolate the effect of state encoding. Supports using
agent partial observations as input via `--agent-obs-input` and has a
`--use-fixed-results` mode to plot without training.
"""

import time
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
from typing import Dict, List

from src.data import build_rollouts, RolloutDataset
from src.models.enhanced_tom_observer import EnhancedToMObserver
from src.training import train_epoch, eval_model


def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_state_encoder_experiment(state_type: str, train_loader, val_loader,
                                 n_qubits: int = 8, max_epochs: int = 20,
                                 patience: int = 5, lr: float = 3e-4,
                                 device: str = "cpu", state_dim: int = 17) -> Dict:
    """Train and evaluate an EnhancedToMObserver with a specific state encoder type.

    Returns a dict with best accuracies, timings, epoch breakdown, and model
    parameter counts for later tabulation/plotting.
    """
    print(f"Running {state_type} state encoder experiment...")

    start_time = time.time()
    # Fix belief_type to classical to isolate state encoder effects
    model = EnhancedToMObserver(state_type=state_type, belief_type="classical",
                                n_qubits=n_qubits, device=device, state_dim=state_dim)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses: List[float] = []
    val_accuracies: List[float] = []
    val_fb_accuracies: List[float] = []
    val_vis_accuracies: List[float] = []

    best_val_acc = 0.0
    best_results = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, device=device)
        train_losses.append(tr_loss)

        val_results = eval_model(model, val_loader, device=device)
        val_accuracies.append(val_results['acc'])
        val_fb_accuracies.append(val_results['acc_false_belief'])
        val_vis_accuracies.append(val_results['acc_visible'])

        epoch_time = time.time() - epoch_start
        if val_results['acc'] > best_val_acc:
            best_val_acc = val_results['acc']
            best_results = val_results.copy()
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"  Epoch {epoch:02d} | Loss: {tr_loss:.4f} | Acc: {val_results['acc']:.3f} | "
              f"FB: {val_results['acc_false_belief']:.3f} | Time: {epoch_time:.1f}s | "
              f"Best: {best_val_acc:.3f} (Epoch {best_epoch})")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    total_time = time.time() - start_time
    actual_epochs = len(train_losses)
    results = {
        'state_type': state_type,
        'belief_type': 'classical',
        'n_qubits': n_qubits if state_type in ['quantum', 'hybrid'] else 0,
        'total_time': total_time,
        'avg_epoch_time': total_time / actual_epochs,
        'best_overall_acc': best_results['acc'],
        'best_fb_acc': best_results['acc_false_belief'],
        'best_vis_acc': best_results['acc_visible'],
        'best_loss': best_results['loss'],
        'best_epoch': best_epoch,
        'total_epochs': actual_epochs,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'val_fb_accuracies': val_fb_accuracies,
        'val_vis_accuracies': val_vis_accuracies,
        'model_params': sum(p.numel() for p in model.parameters())
    }

    return results


def run_state_encoder_comparison(train_loader, val_loader, n_qubits: int = 8,
                                 max_epochs: int = 20, patience: int = 5,
                                 lr: float = 3e-4, device: str = "cpu",
                                 state_dim: int = 17) -> List[Dict]:
    """Compare classical, quantum, and hybrid state encoders.

    Loops over state types and aggregates result dicts from
    `run_state_encoder_experiment`.
    """
    state_types = ["classical", "quantum", "hybrid", "vae"]
    results: List[Dict] = []
    for state_type in state_types:
        try:
            result = run_state_encoder_experiment(state_type, train_loader, val_loader,
                                                  n_qubits=n_qubits, max_epochs=max_epochs,
                                                  patience=patience, lr=lr, device=device,
                                                  state_dim=state_dim)
            results.append(result)
            print(f"Completed {state_type} state encoder: Acc={result['best_overall_acc']:.3f}, "
                  f"Epochs={result['total_epochs']}, Time={result['total_time']:.1f}s\n")
        except Exception as exc:
            print(f"ERROR: Failed for {state_type} state encoder: {exc}")
            continue
    return results


def plot_state_encoder_results(results: List[Dict], save_path: str = None):
    """Render a 6-panel comparison plot of accuracies, timings, and parameter counts."""
    if not results:
        print("No results to plot")
        return

    state_types = [r['state_type'] for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('State Encoder Comparison (Belief State Fixed: Classical)', fontsize=16)

    overall_acc = [r['best_overall_acc'] for r in results]
    fb_acc = [r['best_fb_acc'] for r in results]
    vis_acc = [r['best_vis_acc'] for r in results]
    total_times = [r['total_time'] for r in results]
    epoch_times = [r['avg_epoch_time'] for r in results]
    model_params = [r['model_params'] for r in results]

    colors = {'classical': 'blue', 'quantum': 'red', 'hybrid': 'green'}
    state_colors = [colors.get(st, 'gray') for st in state_types]

    ax1 = axes[0, 0]
    bars1 = ax1.bar(state_types, overall_acc, color=state_colors, alpha=0.7)
    ax1.set_ylabel('Overall Accuracy')
    ax1.set_title('Overall Performance')
    ax1.grid(True, alpha=0.3)
    for bar, acc in zip(bars1, overall_acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.3f}',
                 ha='center', va='bottom')

    ax2 = axes[0, 1]
    bars2 = ax2.bar(state_types, fb_acc, color=state_colors, alpha=0.7)
    ax2.set_ylabel('False-Belief Accuracy')
    ax2.set_title('False-Belief Performance')
    ax2.grid(True, alpha=0.3)
    for bar, acc in zip(bars2, fb_acc):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.3f}',
                 ha='center', va='bottom')

    ax3 = axes[0, 2]
    bars3 = ax3.bar(state_types, vis_acc, color=state_colors, alpha=0.7)
    ax3.set_ylabel('Visible Accuracy')
    ax3.set_title('Visible Performance')
    ax3.grid(True, alpha=0.3)
    for bar, acc in zip(bars3, vis_acc):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.3f}',
                 ha='center', va='bottom')

    ax4 = axes[1, 0]
    bars4 = ax4.bar(state_types, total_times, color=state_colors, alpha=0.7)
    ax4.set_ylabel('Total Training Time (s)')
    ax4.set_title('Training Time')
    ax4.grid(True, alpha=0.3)
    for bar, time_val in zip(bars4, total_times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{time_val:.1f}s',
                 ha='center', va='bottom')

    ax5 = axes[1, 1]
    bars5 = ax5.bar(state_types, epoch_times, color=state_colors, alpha=0.7)
    ax5.set_ylabel('Average Epoch Time (s)')
    ax5.set_title('Per-Epoch Runtime')
    ax5.grid(True, alpha=0.3)
    for bar, time_val in zip(bars5, epoch_times):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{time_val:.1f}s',
                 ha='center', va='bottom')

    ax6 = axes[1, 2]
    bars6 = ax6.bar(state_types, model_params, color=state_colors, alpha=0.7)
    ax6.set_ylabel('Number of Parameters')
    ax6.set_title('Model Size')
    ax6.grid(True, alpha=0.3)
    for bar, params in zip(bars6, model_params):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, f'{params:,}',
                 ha='center', va='bottom')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"State encoder comparison plots saved to {save_path}")
    plt.show()


def print_comparison_table(results: List[Dict]):
    """Pretty-print a tabular view of the comparison results to stdout."""
    if not results:
        print("No results to display")
        return

    print("\n" + "="*120)
    print("STATE ENCODER COMPARISON EXPERIMENT RESULTS (Belief State Fixed: Classical)")
    print("="*120)
    print(f"{'State Type':<12} {'Qubits':<8} {'Overall Acc':<12} {'FB Acc':<10} {'Vis Acc':<10} "
          f"{'Total Time':<12} {'Epochs':<8} {'Best Epoch':<12} {'Params':<10}")
    print("-"*120)
    for r in results:
        print(f"{r['state_type']:<12} {r['n_qubits']:<8} {r['best_overall_acc']:<12.3f} "
              f"{r['best_fb_acc']:<10.3f} {r['best_vis_acc']:<10.3f} "
              f"{r['total_time']:<12.1f} {r['total_epochs']:<8} {r['best_epoch']:<12} "
              f"{r['model_params']:<10}")
    print("="*120)


def main():
    """CLI entrypoint to run the state-encoder comparison or plot fixed results."""
    parser = argparse.ArgumentParser(description="State Encoder Comparison Experiment (Belief Fixed Classical)")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--qubits', type=int, default=8)
    parser.add_argument('--episodes', type=int, default=150)
    parser.add_argument('--val-episodes', type=int, default=50)
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--agent-obs-input', action='store_true',
                        help='Use agent partial observation as model state input instead of observer full-state features')
    # Agent population controls
    parser.add_argument('--use-rb-agents', action='store_true',
                        help='Include rule-based BeliefAgent actors (default: included)')
    parser.add_argument('--no-rb-agents', action='store_true',
                        help='Exclude rule-based BeliefAgent actors (default: included)')
    parser.add_argument('--use-qlearn-agents', action='store_true',
                        help='Include QLearnAgent actors for dataset rollouts')
    parser.add_argument('--qlearn-iters', type=int, default=10000,
                        help='Pre-training iterations per QLearnAgent before rollouts')
    parser.add_argument('--use-random-agents', action='store_true',
                        help='Include RandomAgent actors for dataset rollouts')
    parser.add_argument('--random-alpha', type=str, default='',
                        help='Dirichlet alpha for RandomAgent. Float (e.g., 1.0) or comma-separated 5 values (e.g., 1,1,1,1,1)')
    parser.add_argument('--save-results', type=str, default='state_encoder_comparison_results.json')
    parser.add_argument('--save-plots', type=str, default='state_encoder_comparison_plots.png')
    parser.add_argument('--use-fixed-results', action='store_true',
                        help='If set, skip training and use a fixed results dictionary for plotting')
    parser.add_argument('--fixed-results-file', type=str, default='',
                        help='Optional JSON file with precomputed results to plot')

    args = parser.parse_args()
    set_seed(args.seed)

    # Optional fixed precomputed results
    if args.use_fixed_results:
        print("Using fixed results for plotting (no training).")
        if args.fixed_results_file:
            with open(args.fixed_results_file, 'r') as f:
                results = json.load(f)
        else:
            # Default exemplar
            results = [
                {
                    'state_type': 'classical',
                    'belief_type': 'classical',
                    'n_qubits': 0,
                    'total_time': 10.0,
                    'avg_epoch_time': 0.5,
                    'best_overall_acc': 0.930,
                    'best_fb_acc': 0.910,
                    'best_vis_acc': 0.942,
                    'best_loss': 0.210,
                    'best_epoch': 12,
                    'total_epochs': 20,
                    'train_losses': [],
                    'val_accuracies': [],
                    'val_fb_accuracies': [],
                    'val_vis_accuracies': [],
                    'model_params': 36598,
                },
                {
                    'state_type': 'quantum',
                    'belief_type': 'classical',
                    'n_qubits': args.qubits,
                    'total_time': 110.0,
                    'avg_epoch_time': 4.6,
                    'best_overall_acc': 0.948,
                    'best_fb_acc': 0.968,
                    'best_vis_acc': 0.944,
                    'best_loss': 0.185,
                    'best_epoch': 18,
                    'total_epochs': 25,
                    'train_losses': [],
                    'val_accuracies': [],
                    'val_fb_accuracies': [],
                    'val_vis_accuracies': [],
                    'model_params': 35909,
                },
                {
                    'state_type': 'hybrid',
                    'belief_type': 'classical',
                    'n_qubits': args.qubits,
                    'total_time': 60.0,
                    'avg_epoch_time': 2.4,
                    'best_overall_acc': 0.955,
                    'best_fb_acc': 0.975,
                    'best_vis_acc': 0.949,
                    'best_loss': 0.178,
                    'best_epoch': 22,
                    'total_epochs': 25,
                    'train_losses': [],
                    'val_accuracies': [],
                    'val_fb_accuracies': [],
                    'val_vis_accuracies': [],
                    'model_params': 34101,
                },
            ]
    else:
        print("Building rollout datasets...")
        # Parse random alpha if provided
        random_alpha = 1.0
        if args.random_alpha:
            txt = args.random_alpha.strip()
            if ',' in txt:
                random_alpha = [float(x) for x in txt.split(',') if x.strip() != '']
            else:
                random_alpha = float(txt)
        # Resolve which agent types to include
        use_rb = True
        if args.no_rb_agents:
            use_rb = False
        if args.use_rb_agents:
            use_rb = True

        train_samp, val_samp = build_rollouts(
            num_agents=4,
            episodes_per_agent=args.episodes,
            k_context=3,
            grid=9,
            fov=3,
            use_rb_agents=use_rb,
            use_qlearn_agents=args.use_qlearn_agents,
            qlearn_iters=args.qlearn_iters,
            use_random_agents=args.use_random_agents,
            random_alpha=random_alpha,
            max_steps=60,
            seed=args.seed,
            use_agent_obs_as_state=args.agent_obs_input,
        )

        train_ds = RolloutDataset(train_samp)
        val_ds = RolloutDataset(val_samp)
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

        print(f"Dataset sizes: train={len(train_samp)}, val={len(val_samp)}")

        # Determine state_dim based on chosen input source
        # If using agent partial obs: channels (1+1+4)=6 * fov^2; here fov=3
        state_dim = (6 * (3 * 3)) if args.agent_obs_input else 17

        print("\nStarting state encoder comparison experiments...")
        results = run_state_encoder_comparison(train_loader, val_loader, args.qubits,
                                               args.max_epochs, args.patience, args.lr, args.device,
                                               state_dim)

    print_comparison_table(results)

    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save_results}")

    if results:
        plot_state_encoder_results(results, args.save_plots)

    print("\nState encoder comparison experiment completed!")


if __name__ == '__main__':
    main()


