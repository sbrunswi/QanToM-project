import argparse
import sys
import os
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from new_experiment1.experiment import run_experiment
from utils import utils


def parse_args():
    parser = argparse.ArgumentParser('For ToM Passive Exp')
    parser.add_argument('--num_epoch', '-e', type=int, default=100)
    parser.add_argument('--past', '-p', type=int, default=1, help='Number of past episodes (num_past)')
    parser.add_argument('--num_agent', '-na', type=int, default=1000)
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--learning_rate', '--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--alpha', '-a', type=float, nargs='+', default=0.01)
    parser.add_argument('--save_freq', '-s', type=int, default=10)
    parser.add_argument('--train_dir', default='none', type=str)
    parser.add_argument('--eval_dir', default='none', type=str)
    parser.add_argument('--device', default='cpu', help="cuda, mps, or cpu")
    parser.add_argument('--use_quantum', action='store_true', help="Use quantum-enhanced PredNetQuantum model")
    parser.add_argument('--n_qubits', type=int, default=4, help="Number of qubits for quantum model")
    parser.add_argument('--n_layers', type=int, default=2, help="Number of layers for quantum model")
    args = parser.parse_args()
    return args


def main(args):
    alpha_val = args.alpha[0] if isinstance(args.alpha, list) else args.alpha
    
    # make folder for experiment
    experiment_folder = utils.make_folder(
        alpha=alpha_val, 
        num_past=args.past, 
        main_experiment=None  # No main experiment, just experiment 1
    )
    
    run_experiment(
        num_epoch=args.num_epoch,
        past=args.past,
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
        n_layers=args.n_layers
    )


if __name__ == '__main__':
    args = parse_args()
    main(args)