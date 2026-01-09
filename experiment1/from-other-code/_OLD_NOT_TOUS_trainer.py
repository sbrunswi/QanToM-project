"""
Training and evaluation functions for Theory of Mind models.

- train_epoch: one training pass computing cross-entropy loss on action labels
- eval_model: aggregate loss/accuracy plus False-Belief vs Visible bucket metrics

False-Belief buckets are determined by the dataset's `swap_hidden` flag (1.0
when a recent swap occurred outside the agent's FOV within a window), enabling
per-bucket accuracy reporting.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_epoch(model, loader, opt, device="cpu"):
    """Train a single epoch.

    Iterates batches from `loader`, moves tensors to device, performs forward,
    computes cross-entropy against `label` (next action), backpropagates, and
    steps the optimizer. Returns average loss over all samples in the epoch.
    """
    model.train()
    tot_loss = 0.0
    tot_n = 0
    ce = nn.CrossEntropyLoss()
    for char, mental, state, label, _ in loader:
        char = char.to(device)
        mental = mental.to(device)
        state = state.to(device)
        label = label.to(device)
        opt.zero_grad()
        logits = model(char, mental, state)
        loss = ce(logits, label)
        loss.backward()
        opt.step()
        tot_loss += float(loss.item())*label.size(0)
        tot_n += label.size(0)
    return tot_loss/tot_n


def eval_model(model, loader, device="cpu"):
    """Evaluate model performance with detailed metrics.

    Returns dict with:
    - loss: average cross-entropy over dataset
    - acc: overall accuracy across all samples
    - acc_false_belief: accuracy where `swap_hidden` > 0.5
    - acc_visible: accuracy where `swap_hidden` <= 0.5
    - n: number of evaluated samples
    """
    model.eval()
    ce = nn.CrossEntropyLoss(reduction='sum')
    tot_loss = 0.0
    tot_n = 0
    correct = 0
    # Buckets for false-belief vs visible
    fb_tot = 0
    fb_correct = 0
    vis_tot = 0
    vis_correct = 0

    with torch.no_grad():
        for char, mental, state, label, swap_hidden in loader:
            char = char.to(device)
            mental = mental.to(device)
            state = state.to(device)
            label = label.to(device)
            logits = model(char, mental, state)
            loss = ce(logits, label)
            tot_loss += float(loss.item())
            tot_n += label.size(0)
            pred = torch.argmax(logits, dim=-1)
            correct += int((pred == label).sum().item())
            # buckets
            mask_fb = (swap_hidden.squeeze(-1) > 0.5)
            mask_vis = ~mask_fb
            if mask_fb.any():
                fb_tot += int(mask_fb.sum().item())
                fb_correct += int((pred[mask_fb] == label[mask_fb]).sum().item())
            if mask_vis.any():
                vis_tot += int(mask_vis.sum().item())
                vis_correct += int((pred[mask_vis] == label[mask_vis]).sum().item())

    results = {
        "loss": tot_loss/max(1, tot_n),
        "acc": correct/max(1, tot_n),
        "acc_false_belief": (fb_correct/max(1, fb_tot)) if fb_tot>0 else float('nan'),
        "acc_visible": (vis_correct/max(1, vis_tot)) if vis_tot>0 else float('nan'),
        "n": tot_n,
    }
    return results
