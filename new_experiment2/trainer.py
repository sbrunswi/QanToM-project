"""
Training and evaluation functions for Theory of Mind models (Experiment 2).

- train_epoch: one training pass computing losses for action, consumption, and SR
- eval_model: aggregate loss/accuracy plus optional visualization data

The model predicts:
- Action probabilities (5 actions)
- Consumption probabilities (4 consumption types)
- Successor representation (SR) for 3 gamma values
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


def cross_entropy_with_soft_label(pred, targ):
    """Cross entropy loss with soft labels (for SR prediction)."""
    return -(targ * pred.log()).sum(dim=-1).mean()


def train_epoch(model, loader, optimizer, device="cpu", num_agent=None):
    """Train a single epoch.

    Iterates batches from `loader`, moves tensors to device, performs forward pass,
    computes losses for action, consumption, and SR, backpropagates,
    and steps the optimizer. Returns average losses and accuracies.

    Args:
        model: PredNet model instance
        loader: DataLoader with batches of (past_traj, curr_state, target_action, 
                target_consume, target_sr, indices)
        optimizer: Optimizer instance
        device: Device to run on
        num_agent: Number of agents (for accuracy calculation)

    Returns:
        dict with keys:
            - total_loss: average total loss
            - action_loss: average action loss
            - consumption_loss: average consumption loss
            - sr_loss: average SR loss
            - action_acc: accuracy (exact match on argmax)
            - consumption_acc: consumption accuracy
    """
    model.train()
    
    criterion_nll = nn.NLLLoss()
    
    tot_loss = 0.0
    a_loss = 0.0
    c_loss = 0.0
    s_loss = 0.0
    action_acc = 0.0
    consumption_acc = 0.0
    tot_n = 0
    
    for batch in loader:
        past_traj, curr_state, target_action, target_consume, target_sr, _ = batch
        
        # Move to device
        past_traj = past_traj.float().to(device)
        curr_state = curr_state.float().to(device)
        target_action = target_action.long().to(device).squeeze(-1)
        target_consume = target_consume.long().to(device).squeeze(-1)
        target_sr = target_sr.float().to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_action, pred_consumption, pred_sr, e_char_2d = model(past_traj, curr_state)
        
        # Compute losses
        action_loss = criterion_nll(pred_action, target_action)
        consumption_loss = criterion_nll(pred_consumption, target_consume)
        sr_loss = cross_entropy_with_soft_label(pred_sr, target_sr.flatten(1, 2))
        
        # Total loss
        loss = action_loss + consumption_loss + sr_loss
        
        # Backward pass
        loss.mean().backward()
        optimizer.step()
        
        # Compute accuracy
        pred_action_ind = torch.argmax(pred_action, dim=-1)
        pred_consumption_ind = torch.argmax(pred_consumption, dim=-1)
        
        correct_action = (pred_action_ind == target_action).sum().item()
        correct_consumption = (pred_consumption_ind == target_consume).sum().item()
        
        # Accumulate metrics
        batch_size = target_action.size(0)
        tot_loss += loss.item() * batch_size
        a_loss += action_loss.item() * batch_size
        c_loss += consumption_loss.item() * batch_size
        s_loss += sr_loss.item() * batch_size
        action_acc += correct_action
        consumption_acc += correct_consumption
        tot_n += batch_size
    
    # Calculate accuracies (use num_agent if provided, otherwise use tot_n)
    n_for_acc = num_agent if num_agent is not None else tot_n
    
    return {
        "total_loss": tot_loss / max(1, tot_n),
        "action_loss": a_loss / max(1, tot_n),
        "consumption_loss": c_loss / max(1, tot_n),
        "sr_loss": s_loss / max(1, tot_n),
        "action_acc": action_acc / max(1, n_for_acc),
        "consumption_acc": consumption_acc / max(1, n_for_acc),
    }


def eval_model(model, loader, device="cpu", is_visualize=False, num_agent=None):
    """Evaluate model performance with detailed metrics.

    Args:
        model: PredNet model instance
        loader: DataLoader with batches of (past_traj, curr_state, target_action,
                target_consume, target_sr, indices)
        device: Device to run on
        is_visualize: If True, includes visualization data in results
        num_agent: Number of agents (for accuracy calculation)

    Returns:
        dict with keys:
            - total_loss: average total loss
            - action_loss: average action loss
            - consumption_loss: average consumption loss
            - sr_loss: average SR loss
            - action_acc: accuracy
            - consumption_acc: consumption accuracy
            - past_traj: (optional) past trajectories for visualization [first 16 samples]
            - curr_state: (optional) current states for visualization [first 16 samples]
            - pred_actions: (optional) predicted action probabilities [first 16 samples]
            - pred_consumption: (optional) predicted consumption probabilities [first 16 samples]
            - pred_sr: (optional) predicted SR [first 16 samples]
            - e_char: (optional) character embeddings [all samples]
            - targ_actions: (optional) target actions [first 16 samples]
            - targ_consumption: (optional) target consumption [first 16 samples]
            - targ_sr: (optional) target SR [first 16 samples]
    """
    model.eval()
    
    criterion_nll = nn.NLLLoss()
    
    tot_loss = 0.0
    a_loss = 0.0
    c_loss = 0.0
    s_loss = 0.0
    action_acc = 0.0
    consumption_acc = 0.0
    tot_n = 0
    
    # Storage for visualization data
    vis_past_traj = None
    vis_curr_state = None
    vis_pred_actions = None
    vis_pred_consumption = None
    vis_pred_sr = None
    vis_targ_actions = None
    vis_targ_consumption = None
    vis_targ_sr = None
    all_e_char = []
    
    with torch.no_grad():
        for batch in loader:
            past_traj, curr_state, target_action, target_consume, target_sr, _ = batch
            
            # Move to device
            past_traj = past_traj.float().to(device)
            curr_state = curr_state.float().to(device)
            target_action = target_action.long().to(device).squeeze(-1)
            target_consume = target_consume.long().to(device).squeeze(-1)
            target_sr = target_sr.float().to(device)
            
            # Forward pass
            pred_action, pred_consumption, pred_sr, e_char = model(past_traj, curr_state)
            
            # Compute losses
            action_loss = criterion_nll(pred_action, target_action)
            consumption_loss = criterion_nll(pred_consumption, target_consume)
            sr_loss = cross_entropy_with_soft_label(pred_sr, target_sr.flatten(1, 2))
            
            # Total loss
            loss = action_loss + consumption_loss + sr_loss
            
            # Compute accuracy
            pred_action_ind = torch.argmax(pred_action, dim=-1)
            pred_consumption_ind = torch.argmax(pred_consumption, dim=-1)
            
            correct_action = (pred_action_ind == target_action).sum().item()
            correct_consumption = (pred_consumption_ind == target_consume).sum().item()
            
            # Accumulate metrics
            batch_size = target_action.size(0)
            tot_loss += loss.item() * batch_size
            a_loss += action_loss.item() * batch_size
            c_loss += consumption_loss.item() * batch_size
            s_loss += sr_loss.item() * batch_size
            action_acc += correct_action
            consumption_acc += correct_consumption
            tot_n += batch_size
            
            # Store visualization data (first batch only, first 16 samples)
            if is_visualize:
                if vis_past_traj is None:
                    n_samples = min(16, batch_size)
                    vis_past_traj = past_traj[:n_samples].cpu().numpy()
                    vis_curr_state = curr_state[:n_samples].cpu().numpy()
                    vis_pred_actions = pred_action[:n_samples].cpu().numpy()
                    vis_pred_consumption = pred_consumption[:n_samples].cpu().numpy()
                    vis_pred_sr = pred_sr[:n_samples].reshape(-1, target_sr.shape[1], 
                                                             target_sr.shape[2], 3).cpu().numpy()
                    
                    # Convert targets to one-hot for visualization
                    diag = torch.eye(5, device=target_action.device)
                    vis_targ_actions = diag[target_action[:n_samples]].cpu().numpy()
                    vis_targ_consumption = diag[target_consume[:n_samples]].cpu().numpy()
                    vis_targ_sr = target_sr[:n_samples].cpu().numpy()
                
                # Store all e_char embeddings
                all_e_char.append(e_char.cpu().numpy())
    
    # Calculate accuracies
    n_for_acc = num_agent if num_agent is not None else tot_n
    
    results = {
        "total_loss": tot_loss / max(1, tot_n),
        "action_loss": a_loss / max(1, tot_n),
        "consumption_loss": c_loss / max(1, tot_n),
        "sr_loss": s_loss / max(1, tot_n),
        "action_acc": action_acc / max(1, n_for_acc),
        "consumption_acc": consumption_acc / max(1, n_for_acc),
    }
    
    # Add visualization data if requested
    if is_visualize:
        results["past_traj"] = vis_past_traj
        results["curr_state"] = vis_curr_state
        results["pred_actions"] = vis_pred_actions
        results["pred_consumption"] = vis_pred_consumption
        results["pred_sr"] = vis_pred_sr
        results["targ_actions"] = vis_targ_actions
        results["targ_consumption"] = vis_targ_consumption
        results["targ_sr"] = vis_targ_sr
        if all_e_char:
            # Concatenate all e_char embeddings
            results["e_char"] = np.concatenate(all_e_char, axis=0)
    
    return results
