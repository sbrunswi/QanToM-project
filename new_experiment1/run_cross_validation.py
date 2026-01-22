"""
Script to run k-fold cross-validation for Theory of Mind experiments.

Usage:
    python run_cross_validation.py --num_folds 5 --alpha 1.0 [other arguments]
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from new_experiment1.cross_validation import run_cross_validation
from utils import utils


def parse_args():
    parser = argparse.ArgumentParser('Cross-validation for ToM Passive Exp')
    parser.add_argument('--num_folds', '-k', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--num_epoch', '-e', type=int, default=100,
                       help='Number of training epochs per fold')
    parser.add_argument('--past', '-p', type=int, default=1,
                       help='Number of past episodes (num_past)')
    parser.add_argument('--num_agent', '-na', type=int, default=1000,
                       help='Total number of agents in the population')
    parser.add_argument('--batch_size', '-b', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', '--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--alpha', '-a', type=float, nargs='+', default=0.01,
                       help='Alpha parameter(s) for agent population')
    parser.add_argument('--save_freq', '-s', type=int, default=10,
                       help='Frequency of checkpoint saving')
    parser.add_argument('--device', default='cpu',
                       help="Device to use: 'cuda', 'mps', or 'cpu'")
    parser.add_argument('--use_quantum', action='store_true',
                       help="Use quantum-enhanced PredNetQuantum model")
    parser.add_argument('--n_qubits', type=int, default=4,
                       help="Number of qubits for quantum model")
    parser.add_argument('--n_layers', type=int, default=2,
                       help="Number of layers for quantum model")
    parser.add_argument('--loss_type', type=str, default='cross_entropy',
                       choices=['cross_entropy', 'kl_divergence'],
                       help="Type of loss function")
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for fold splitting')
    args = parser.parse_args()
    return args


def main(args):
    try:
        alpha_val = args.alpha[0] if isinstance(args.alpha, list) else args.alpha
        
        print(f"\n{'='*80}")
        print("CROSS-VALIDATION EXPERIMENT")
        print(f"{'='*80}")
        print(f"Alpha: {alpha_val}")
        print(f"Number of folds: {args.num_folds}")
        print(f"Number of agents: {args.num_agent}")
        print(f"Epochs per fold: {args.num_epoch}")
        print(f"Past episodes: {args.past}")
        print(f"{'='*80}\n")
        
        # Make folder for cross-validation experiment
        experiment_folder = utils.make_folder(
            alpha=alpha_val,
            num_past=args.past,
            main_experiment=None
        )
        
        # Add CV suffix to folder name
        cv_suffix = f'_cv_{args.num_folds}folds'
        experiment_folder = experiment_folder + cv_suffix
        os.makedirs(experiment_folder, exist_ok=True)
        
        print(f"Experiment folder: {experiment_folder}\n")
        
        # Run cross-validation
        results = run_cross_validation(
            num_folds=args.num_folds,
            num_epoch=args.num_epoch,
            past=args.past,
            num_agent=args.num_agent,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            experiment_folder=experiment_folder,
            alpha=args.alpha,
            save_freq=args.save_freq,
            train_dir='none',
            eval_dir='none',
            device=args.device,
            use_quantum=args.use_quantum,
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            loss_type=args.loss_type,
            random_state=args.random_state,
            verbose=True
        )
        
        print(f"\nCross-validation completed!")
        print(f"Results saved to: {experiment_folder}")
        print(f"\nFinal Results:")
        print(f"  Mean Validation Accuracy: {results['mean_val_acc']:.4f} ± {results['std_val_acc']:.4f}")
        print(f"  Mean Validation Loss:     {results['mean_val_loss']:.4f} ± {results['std_val_loss']:.4f}")
    
    except KeyboardInterrupt:
        print("\n\nCross-validation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Cross-validation failed with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    args = parse_args()
    main(args)
