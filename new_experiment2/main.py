import argparse
import sys
import os
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from new_experiment2.experiment import run_experiment
from utils import utils


def parse_args():
    parser = argparse.ArgumentParser('For ToM Experiment 2')
    parser.add_argument('--num_epoch', '-e', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--past', '-p', type=int, default=1,
                       help='Number of past episodes (num_past)')
    parser.add_argument('--num_step', type=int, default=31,
                       help='Number of steps per episode (num_step)')
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
    parser.add_argument('--alpha', '-a', type=float, nargs='+', default=0.01,
                       help='Alpha parameter(s) for agent population')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Frequency of checkpoint saving')
    parser.add_argument('--train_dir', default='none', type=str,
                       help='Directory for training data (default: generate on fly)')
    parser.add_argument('--eval_dir', default='none', type=str,
                       help='Directory for eval data (default: generate on fly)')
    parser.add_argument('--device', default='cpu',
                       help="Device to use: 'cuda', 'mps', or 'cpu'")
    parser.add_argument('--use_quantum', action='store_true',
                       help="Use quantum-enhanced PredNetQuantum model")
    parser.add_argument('--n_qubits', type=int, default=3,
                       help="Number of qubits for quantum model")
    parser.add_argument('--n_layers', type=int, default=2,
                       help="Number of layers for quantum model")
    args = parser.parse_args()
    return args


def main(args):
    alpha_val = args.alpha[0] if isinstance(args.alpha, list) else args.alpha

    # Make folder for experiment
    experiment_folder = utils.make_folder(
        alpha=alpha_val,
        num_past=args.past,  # Use past argument for num_past
        main_experiment=2  # Experiment 2
    )

    run_experiment(
        num_epoch=args.num_epoch,
        num_step=args.num_step,
        move_penalty=args.move_penalty,
        height=args.height,
        width=args.width,
        num_agent=args.num_agent,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        experiment_folder=experiment_folder,
        alpha=args.alpha,
        save_freq=args.save_freq,
        train_dir=args.train_dir,
        eval_dir=args.eval_dir,
        device=args.device,
        use_quantum=args.use_quantum,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        num_past=args.past
    )


if __name__ == '__main__':
    args = parse_args()
    main(args)
