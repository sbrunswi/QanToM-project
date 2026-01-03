from environment.env import GridWorldEnv
from experiment1 import model
from experiment1.store_trajectories import Storage
from experiment1.config import get_configs

from utils.visualize import *
from utils import utils
from utils import dataset
from utils import writer

from torch.utils.data import DataLoader
import torch as tr
import torch.optim as optim
import numpy as np
import os


def train(tom_net, optimizer, train_loader, eval_loader, experiment_folder, writer, dicts, visualizer=None):

    # Track metrics for plotting
    train_losses = []
    train_accs = []
    eval_losses = []
    eval_accs = []
    epochs = []

    for epoch in range(dicts['num_epoch']):
        results = tom_net.train(train_loader, optimizer)

        ev_results = evaluate(tom_net, eval_loader)

        if epoch % dicts['save_freq'] == 0:
            utils.save_model(tom_net, dicts, experiment_folder, epoch)
        writer.write(results, epoch, is_train=True)
        writer.write(ev_results, epoch, is_train=False)
        print('Train| Epoch {} Loss |{:.4f}|Acc |{:.4f}'.format(epoch, results['action_loss'], results['action_acc']))
        print('Eval| Epoch {} Loss |{:.4f}|Acc |{:.4f}'.format(epoch, ev_results['action_loss'], ev_results['action_acc']))

        # Track metrics for plotting
        train_losses.append(results['action_loss'])
        train_accs.append(results['action_acc'])
        eval_losses.append(ev_results['action_loss'])
        eval_accs.append(ev_results['action_acc'])
        epochs.append(epoch)

    # Generate loss curves plot after training
    if visualizer is not None:
        visualizer.plot_loss_curves(train_losses, train_accs, eval_losses, eval_accs, epochs)



def evaluate(tom_net, eval_loader, visualizer=None, is_visualize=False,
             most_act=None, count_act=None):
    '''
    we provide the base result of figure 2,
    but if you want to show the other results,
    run the inference.py after you have the models.
    '''
    with tr.no_grad():
        ev_results = tom_net.evaluate(eval_loader, is_visualize=is_visualize)

    if is_visualize:
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
    return ev_results

def run_experiment(num_epoch, main_experiment, sub_experiment, num_agent, batch_size, lr,
                   experiment_folder, alpha, save_freq,train_dir='none', eval_dir='none', device=None):

    exp_kwargs, env_kwargs, model_kwargs, agent_kwargs = get_configs(sub_experiment)
    population = utils.make_pool('random', exp_kwargs['move_penalty'], alpha, num_agent)
    env = GridWorldEnv(env_kwargs)
    model_kwargs['device'] = device
    tom_net = model.PredNet(**model_kwargs)
    tom_net.to(device)
    # if model_kwargs['device'] == 'cuda':
    #     tom_net = tom_net.cuda()
    # else

    dicts = dict(main=main_experiment, sub=sub_experiment, alpha=alpha, batch_size=batch_size,
                 lr=lr, num_epoch=num_epoch, save_freq=save_freq)

    # Make the Dataset
    train_storage = Storage(env, population, exp_kwargs['num_past'], exp_kwargs['num_step'])
    eval_storage = Storage(env, population, exp_kwargs['num_past'], exp_kwargs['num_step'])
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

    # Train
    optimizer = optim.Adam(tom_net.parameters(), lr=lr)
    train(tom_net, optimizer, train_loader, eval_loader, experiment_folder,
          summary_writer, dicts, visualizer=visualizer)

    # Save final checkpoint (after training completes)
    final_epoch = dicts['num_epoch'] - 1
    utils.save_model(tom_net, dicts, experiment_folder, final_epoch)

    # Test
    eval_storage.reset()
    test_data = eval_storage.extract()
    test_data['exp'] = 'exp1'
    test_dataset = dataset.ToMDataset(**test_data)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    most_act, count_act = eval_storage.get_most_act()
    ev_results = evaluate(tom_net, test_loader, visualizer, is_visualize=True,
                          most_act=most_act, count_act=count_act)
    
    return ev_results  # Return final evaluation results



