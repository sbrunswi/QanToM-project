
def get_configs(num_past, max_step=1):
    """
    Get configuration parameters for the experiment.
    
    Args:
        num_past: Number of past episodes to use
        max_step: Maximum number of steps per episode
    
    Returns:
        exp_kwargs, env_kwargs, model_kwargs, agent_kwargs
    """
    num_step = max_step
    move_penalty = -0.01

    env_kwargs = dict(height=11, width=11, pixel_per_grid=8, preference=100, exp=1, save=True)
    model_kwargs = dict(num_past=num_past, num_input=11, device='cuda')
    exp_kwargs = dict(num_past=num_past, num_step=num_step, move_penalty=move_penalty)
    agent_kwargs = dict(agent_type='random')

    return exp_kwargs, env_kwargs, model_kwargs, agent_kwargs

#def get_inference_configs(num_exp):
