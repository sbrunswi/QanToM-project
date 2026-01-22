"""
Training and evaluation functions for Theory of Mind models.

- train_epoch: one training pass computing cross entropy loss on action probabilities
- eval_model: aggregate loss/accuracy plus optional visualization data

The model predicts action probabilities (5 actions) from past trajectories and current state.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


def train_epoch(model, loader, optimizer, device="cpu", loss_type="cross_entropy"):
    """Train a single epoch.

    Iterates batches from `loader`, moves tensors to device, performs forward pass,
    computes loss against target action classes, backpropagates,
    and steps the optimizer. Returns average loss and accuracy over all samples.

    Args:
        model: PredNet model instance
        loader: DataLoader with batches of (past_traj, curr_state, target_action)
        optimizer: Optimizer instance
        device: Device to run on
        loss_type: Type of loss ("cross_entropy" or "kl_divergence")

    Returns:
        dict with keys:
            - action_loss: average loss
            - action_acc: accuracy (exact match on argmax)
    """
    model.train()
    
    # Choose loss function
    if loss_type == "cross_entropy":
        criterion = nn.NLLLoss()
    elif loss_type == "kl_divergence":
        criterion = nn.KLDivLoss(reduction='batchmean')
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Must be 'cross_entropy' or 'kl_divergence'")
    
    tot_loss = 0.0
    tot_acc = 0.0
    tot_n = 0
    
    for batch in loader:
        past_traj, curr_state, target = batch
        
        # Move to device
        past_traj = past_traj.to(dtype=torch.float, device=device)
        curr_state = curr_state.to(dtype=torch.float, device=device)
        target = target.to(dtype=torch.float, device=device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred, _ = model(past_traj, curr_state)
        pred = pred.clamp(min=1e-8)  # Avoid numerical issues
        
        # Compute loss based on loss_type
        if loss_type == "cross_entropy":
            # Convert one-hot target to class indices for CrossEntropyLoss
            target_classes = torch.argmax(target, dim=-1).long()
            loss = criterion(torch.log(pred), target_classes)
        elif loss_type == "kl_divergence":
            # KL divergence: pred should be log-probabilities, target should be probabilities
            loss = criterion(torch.log(pred), target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        pred_onehot = torch.argmax(pred, dim=-1)
        targ_onehot = torch.argmax(target, dim=-1)
        correct = (pred_onehot == targ_onehot).sum().item()
        
        # Accumulate metrics
        batch_size = target.size(0)
        tot_loss += loss.item() * batch_size
        tot_acc += correct
        tot_n += batch_size
    
    return {
        "action_loss": tot_loss / max(1, tot_n),
        "action_acc": tot_acc / max(1, tot_n),
    }


def eval_model(model, loader, device="cpu", is_visualize=False, loss_type="cross_entropy"):
    """Evaluate model performance with detailed metrics.

    Args:
        model: PredNet model instance
        loader: DataLoader with batches of (past_traj, curr_state, target_action)
        device: Device to run on
        is_visualize: If True, includes visualization data in results
        loss_type: Type of loss ("cross_entropy" or "kl_divergence")

    Returns:
        dict with keys:
            - action_loss: average loss
            - action_acc: accuracy (exact match on argmax)
            - past_traj: (optional) past trajectories for visualization [first 16 samples]
            - curr_state: (optional) current states for visualization [first 16 samples]
            - pred_actions: (optional) predicted action probabilities [first 16 samples]
            - e_char: (optional) character embeddings [all samples]
    """
    model.eval()
    
    # Choose loss function
    if loss_type == "cross_entropy":
        criterion = nn.NLLLoss(reduction='sum')
    elif loss_type == "kl_divergence":
        criterion = nn.KLDivLoss(reduction='sum')
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Must be 'cross_entropy' or 'kl_divergence'")
    
    tot_loss = 0.0
    tot_acc = 0.0
    tot_n = 0
    
    # Storage for visualization data
    vis_past_traj = None
    vis_curr_state = None
    vis_pred_actions = None
    all_e_char = []
    
    with torch.no_grad():
        for batch in loader:
            past_traj, curr_state, target = batch
            
            # Move to device
            past_traj = past_traj.to(dtype=torch.float, device=device)
            curr_state = curr_state.to(dtype=torch.float, device=device)
            target = target.to(dtype=torch.float, device=device)
            
            # Forward pass
            pred, e_char = model(past_traj, curr_state)
            pred = pred.clamp(min=1e-8)  # Avoid log(0)
            
            # Compute loss based on loss_type
            if loss_type == "cross_entropy":
                # Convert one-hot target to class indices for CrossEntropyLoss
                target_classes = torch.argmax(target, dim=-1).long()
                loss = criterion(torch.log(pred), target_classes)
            elif loss_type == "kl_divergence":
                # KL divergence: pred should be log-probabilities, target should be probabilities
                loss = criterion(torch.log(pred), target)
            
            # Compute accuracy
            pred_onehot = torch.argmax(pred, dim=-1)
            targ_onehot = torch.argmax(target, dim=-1)
            correct = (pred_onehot == targ_onehot).sum().item()
            
            # Accumulate metrics
            batch_size = target.size(0)
            tot_loss += loss.item()
            tot_acc += correct
            tot_n += batch_size
            
            # Store visualization data (first batch only, first 16 samples)
            if is_visualize:
                if vis_past_traj is None:
                    n_samples = min(16, batch_size)
                    vis_past_traj = past_traj[:n_samples].cpu().numpy()
                    vis_curr_state = curr_state[:n_samples].cpu().numpy()
                    vis_pred_actions = pred[:n_samples].cpu().numpy()
                
                # Store all e_char embeddings
                all_e_char.append(e_char.cpu().numpy())
    
    results = {
        "action_loss": tot_loss / max(1, tot_n),
        "action_acc": tot_acc / max(1, tot_n),
    }
    
    # Add visualization data if requested
    if is_visualize:
        results["past_traj"] = vis_past_traj
        results["curr_state"] = vis_curr_state
        results["pred_actions"] = vis_pred_actions
        if all_e_char:
            # Concatenate all e_char embeddings
            results["e_char"] = np.concatenate(all_e_char, axis=0)
    
    return results

