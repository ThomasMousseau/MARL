import matplotlib.pyplot as plt
import matplotlib.animation as animation

from vmas.simulator.scenario import BaseScenario
from typing import Union
import time
import torch
from vmas import make_env
from vmas.simulator.core import Agent

from torch import nn

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.fc(x)

def train_policy(env, policy, optimizer, num_episodes, gamma=0.99):
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_rewards = []
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action_probs = policy(obs_tensor)
            action = torch.multinomial(action_probs, 1).item()
            next_obs, reward, done, _ = env.step(action)
            episode_rewards.append(reward)
            obs = next_obs

        # Compute discounted rewards and update policy
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(episode_rewards):
            cumulative_reward = reward + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        #* REINFORCE algorithm 
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        optimizer.zero_grad()
        loss = -torch.sum(torch.log(action_probs) * discounted_rewards)
        loss.backward()
        optimizer.step()

def _get_deterministic_action(agent: Agent, continuous: bool, env):
    if continuous:
        action = -agent.action.u_range_tensor.expand(env.batch_dim, agent.action_size)
    else:
        action = (
            torch.tensor([1], device=env.device, dtype=torch.long)
            .unsqueeze(-1)
            .expand(env.batch_dim, 1)
        )
    return action.clone()

def _get_custom_action(agent, env):
    # Replace this with your own agent's policy
    # For example, a simple random policy:
    return env.get_random_action(agent)

def use_vmas_env(
    render: bool,
    num_envs: int,
    n_steps: int,
    device: str,
    scenario: Union[str, BaseScenario],
    continuous_actions: bool,
    random_action: bool,
    **kwargs
):
    """Example function to use a vmas environment.

    This is a simplification of the function in `vmas.examples.use_vmas_env.py`.

    Args:
        continuous_actions (bool): Whether the agents have continuous or discrete actions
        scenario (str): Name of scenario
        device (str): Torch device to use
        render (bool): Whether to render the scenario\num_envs (int): Number of vectorized environments
        n_steps (int): Number of steps before returning done
        random_action (bool): Use random actions or have all agents perform the down action

    """

    scenario_name = scenario if isinstance(scenario,str) else scenario.__class__.__name__

    env = make_env(
        scenario=scenario,
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        seed=0,
        # Environment specific variables
        **kwargs
    )

    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0

    for s in range(n_steps):
        step += 1
        print(f"Step {step}")

        actions = []
        for i, agent in enumerate(env.agents):
            if not random_action:
                action = _get_custom_action(agent, env)
                #action = _get_deterministic_action(agent, continuous_actions, env)
            else:
                action = env.get_random_action(agent)

            actions.append(action)

        obs, rews, dones, info = env.step(actions)

        if render:
            frame = env.render(
                mode="rgb_array",
                agent_index_focus=None,  # Can give the camera an agent index to focus on
            )
            frame_list.append(frame)

    total_time = time.time() - init_time
    print(
        f"It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device} "
        f"for {scenario_name} scenario."
    )

    if render:
        from moviepy import ImageSequenceClip
        fps=30
        clip = ImageSequenceClip(frame_list, fps=fps)
        clip.write_gif(f'{scenario_name}.gif', fps=fps)
    
    from IPython.display import Image
    Image(f'{scenario_name}.gif')
    

if __name__ == "__main__":
    scenario_name = "football"
    use_vmas_env(
        scenario=scenario_name,
        render=True,
        num_envs=32,
        n_steps=1000,
        device="cpu",
        continuous_actions=False,
        random_action=False,
        # Environment specific variables
        ai_red_agents=False,
        ai_blue_agents=False,
        n_blue_agents=2,
        n_red_agents=2,
        max_speed=0.5,
        ball_max_speed=1.0
    )


