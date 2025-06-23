# Proximal Policy Optimization (PPO) Algorithm Implementation
# This script implements the PPO algorithm for discrete action spaces using PyTorch.
# It includes Actor and Critic networks, a PPO agent class, and a replay buffer (Memory).
# An example usage with the CartPole-v1 environment from Gymnasium is provided in the main section.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym # Using gymnasium instead of gym for up-to-date API
import numpy as np

# Actor Network
class Actor(nn.Module):
    """
    Actor network (policy) for PPO.
    Outputs a probability distribution over actions.
    """
    def __init__(self, state_dim: int, action_dim: int):
        """
        Initializes the Actor network.
        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, action_dim) # Output layer providing action probabilities

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the actor network.
        Args:
            state (torch.Tensor): The current state.
        Returns:
            torch.Tensor: Action probabilities.
        """
        x = F.relu(self.fc1(state))
        action_probs = F.softmax(self.fc2(x), dim=-1) # Softmax for discrete action probabilities
        return action_probs

# Critic Network
class Critic(nn.Module):
    """
    Critic network (value function) for PPO.
    Outputs the value of a given state.
    """
    def __init__(self, state_dim: int):
        """
        Initializes the Critic network.
        Args:
            state_dim (int): Dimension of the state space.
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 1)          # Output layer providing the state value

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the critic network.
        Args:
            state (torch.Tensor): The current state.
        Returns:
            torch.Tensor: The estimated value of the state.
        """
        x = F.relu(self.fc1(state))
        value = self.fc2(x)
        return value

class PPO:
    """
    Proximal Policy Optimization (PPO) agent.
    Handles action selection, policy updates, and network management.
    """
    def __init__(self, state_dim: int, action_dim: int, lr_actor: float, lr_critic: float,
                 gamma: float, K_epochs: int, eps_clip: float, device: torch.device):
        """
        Initializes the PPO agent.
        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            lr_actor (float): Learning rate for the actor network.
            lr_critic (float): Learning rate for the critic network.
            gamma (float): Discount factor for future rewards.
            K_epochs (int): Number of epochs to update the policy per PPO update cycle.
            eps_clip (float): Clipping parameter for the PPO objective function.
            device (torch.device): Device to run the computations on (e.g., 'cpu' or 'cuda').
        """
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        # Initialize actor and critic networks
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Initialize old actor network for PPO updates (target network)
        self.actor_old = Actor(state_dim, action_dim).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict()) # Copy weights from current actor

        self.MseLoss = nn.MSELoss() # Mean Squared Error loss for critic updates

    def select_action(self, state: np.ndarray) -> tuple[int, float]:
        """
        Selects an action based on the current state using the old actor policy.
        Args:
            state (np.ndarray): The current state of the environment.
        Returns:
            tuple[int, float]: The selected action and its log probability.
        """
        with torch.no_grad(): # No gradient calculation needed for action selection
            state_tensor = torch.FloatTensor(state).to(self.device)
            action_probs = self.actor_old(state_tensor)
            dist = Categorical(action_probs) # Create a categorical distribution over actions
            action = dist.sample() # Sample an action
            action_logprob = dist.log_prob(action) # Get the log probability of the sampled action
        return action.item(), action_logprob.cpu().numpy()

    def update(self, memory: 'Memory'):
        """
        Updates the actor and critic networks using data collected in the memory buffer.
        Args:
            memory (Memory): The replay buffer containing trajectories.
        """
        # Monte Carlo estimate of returns (discounted rewards)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal: # If the episode ended at this step
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward) # Insert at the beginning to maintain order

        # Normalizing the rewards (helps stabilize training)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-7) # Add epsilon for numerical stability

        # Convert lists to tensors
        # .stack converts list of tensors to a single tensor.
        # .squeeze removes dimensions of size 1.
        # .detach() prevents gradient flow to these tensors as they are considered fixed targets.
        old_states_tensor = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(self.device)
        old_actions_tensor = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(self.device)
        old_logprobs_tensor = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(self.device)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values using the current policy (actor and critic)
            logprobs, state_values, dist_entropy = self.evaluate(old_states_tensor, old_actions_tensor)

            # Match state_values tensor dimensions with rewards tensor (squeeze if necessary)
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta_old):
            # This is a core part of PPO, measuring how much the new policy differs from the old one.
            ratios = torch.exp(logprobs - old_logprobs_tensor.detach())

            # Finding Surrogate Loss:
            # Advantages: how much better an action is compared to the average action at that state.
            advantages = rewards_tensor - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Final loss of clipped objective PPO:
            # Actor loss: -min(surr1, surr2) - encourages policy improvement while penalizing large changes.
            # Critic loss: 0.5 * MseLoss(state_values, rewards_tensor) - trains the critic to predict returns.
            # Entropy bonus: -0.01 * dist_entropy - encourages exploration.
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards_tensor) - 0.01 * dist_entropy

            # Take gradient step for both actor and critic
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            loss.mean().backward() # Calculate gradients for the mean loss over the batch
            self.optimizer_actor.step()
            self.optimizer_critic.step()

        # Copy new weights into old policy (actor_old) after all K_epochs of updates
        self.actor_old.load_state_dict(self.actor.state_dict())

        # Clear memory buffer after update
        memory.clear()

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates the given state-action pairs using the current actor and critic networks.
        Used during the PPO update step.
        Args:
            state (torch.Tensor): Batch of states.
            action (torch.Tensor): Batch of actions taken in those states.
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Log probabilities of the actions under the current policy.
                - State values predicted by the critic.
                - Entropy of the action distribution.
        """
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action) # Log prob of actions taken
        dist_entropy = dist.entropy()           # Entropy of the policy distribution
        state_values = self.critic(state)       # Value of the states
        return action_logprobs, state_values, dist_entropy

# Simple Buffer to store transitions (trajectories)
class Memory:
    """
    A simple replay buffer to store trajectories (state, action, logprob, reward, done).
    """
    def __init__(self):
        self.actions = []       # List to store actions taken
        self.states = []        # List to store states encountered
        self.logprobs = []      # List to store log probabilities of actions
        self.rewards = []       # List to store rewards received
        self.is_terminals = []  # List to store done/terminated flags

    def clear(self):
        """Clears all stored trajectories from the buffer."""
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

def main():
    """
    Main function to set up the PPO agent, environment, and run the training loop.
    """
    ############## Hyperparameters ##############
    env_name = "CartPole-v1"    # Name of the Gymnasium environment
    # creating environment
    env = gym.make(env_name)    # Create the environment instance
    state_dim = env.observation_space.shape[0] # Get state dimension
    action_dim = env.action_space.n            # Get action dimension

    render = False              # Whether to render the environment during training
    # Quick test parameters
    solved_reward = 475         # Target average reward to consider the environment solved (for CartPole-v1, often around 475-500 for stricter criteria)
    log_interval = 20           # Print average reward every 'log_interval' episodes
    max_episodes = 5000         # Maximum number of training episodes
    max_timesteps = 300         # Maximum timesteps per episode (CartPole default is 500, can be adjusted)

    update_timestep = 2000      # Update policy every 'update_timestep' timesteps collected
    K_epochs = 40               # Number of epochs to update policy per PPO update cycle (can be tuned)
    eps_clip = 0.2              # Clipping parameter for PPO
    gamma = 0.99                # Discount factor for future rewards

    lr_actor = 0.0003           # Learning rate for actor network
    lr_critic = 0.001           # Learning rate for critic network

    # Set device for PyTorch (CPU or GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #############################################

    # Initialize PPO agent and memory buffer
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device)
    memory = Memory()

    print(f"Training PPO agent on {env_name} using device: {device}")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")

    # Logging variables
    running_reward = 0      # Accumulated reward for logging interval
    avg_length = 0          # Average episode length for logging interval
    time_step = 0           # Counter for total timesteps collected across all episodes

    # Training loop
    for i_episode in range(1, max_episodes + 1):
        state, _ = env.reset() # Reset environment at the start of each episode
        episode_reward = 0
        for t in range(max_timesteps):
            time_step += 1
            # Agent selects action based on current state
            action, action_logprob = ppo_agent.select_action(state)

            # Store data in memory
            # Ensure data stored are tensors on the correct device if they will be stacked later
            memory.logprobs.append(torch.tensor(action_logprob, dtype=torch.float32).to(device))
            memory.states.append(torch.FloatTensor(state).to(device)) # Convert state to FloatTensor
            memory.actions.append(torch.tensor(action, dtype=torch.long).to(device)) # Action is discrete

            # Perform action in the environment
            state, reward, done, truncated, _ = env.step(action)

            # Store reward and done flag
            memory.rewards.append(reward)
            memory.is_terminals.append(done or truncated) # Episode ends if done or truncated

            episode_reward += reward

            # Update PPO agent if enough timesteps have been collected
            if time_step % update_timestep == 0:
                print(f"Episode {i_episode}: Updating policy at timestep {time_step}...")
                ppo_agent.update(memory)
                # Memory is cleared inside ppo_agent.update()

            if render:
                env.render() # Render the environment if specified

            if done or truncated: # End episode if terminated or truncated
                break

        running_reward += episode_reward
        avg_length += (t + 1) # t is 0-indexed, so length is t+1

        # Log results and check for solved condition
        if i_episode % log_interval == 0:
            current_avg_length = int(avg_length / log_interval)
            current_avg_reward = running_reward / log_interval

            print(f'Episode {i_episode} \t Avg length: {current_avg_length} \t Avg reward: {current_avg_reward:.2f}')

            # Check if solved
            if current_avg_reward >= solved_reward:
                print(f"########## Solved! Average reward is {current_avg_reward:.2f} over last {log_interval} episodes ##########")
                # Example: Save the trained model
                # torch.save(ppo_agent.actor_old.state_dict(), f'./PPO_{env_name}_solved.pth')
                break # Stop training

            running_reward = 0 # Reset running reward for the next logging interval
            avg_length = 0     # Reset average length for the next logging interval

        # Optional: Save model periodically
        # if i_episode % 500 == 0:
            # print(f"Saving model at episode {i_episode}...")
            # torch.save(ppo_agent.actor_old.state_dict(), f'./PPO_{env_name}_episode_{i_episode}.pth')

    env.close() # Close the environment when training is done
    print("Training finished.")

if __name__ == "__main__":
    main()
