import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)  # Second fully connected layer
        self.fc3 = nn.Linear(64, action_size)  # Output layer (one output for each possible action)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation after the first layer
        x = torch.relu(self.fc2(x))  # Apply ReLU after the second layer
        x = torch.softmax(self.fc3(x) - self.fc3(x).max(dim=-1, keepdim=True)[0], dim=-1) # Apply softmax to get action probabilities
        return x
    
def train_policy_network(env, policy_net, optimizer, episodes=1000, gamma=0.99):
    all_rewards = []
    for episode in range(episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)  # Reset environment and convert to tensor
        log_probs = []  # Store the log probabilities of actions taken
        rewards = []  # Store rewards received at each step
        while not env.done:
            # Forward pass: Get action probabilities
            action_probs = policy_net(state)
            action_dist = torch.distributions.Categorical(action_probs)  # Categorical distribution over actions
            
            # Sample an action (choose which question to ask)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)  # Log probability of the action
            log_probs.append(log_prob)
            # Execute the action in the environment
            next_state, reward, done = env.step(action.item())
            rewards.append(reward)  # Store the reward
            # Update the state
            state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        # Compute the discounted rewards
        discounted_rewards = []
        cumulative_reward = 0
        for r in reversed(rewards):
            cumulative_reward = r + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        # Convert to tensor
        discounted_rewards = torch.tensor(discounted_rewards)

        # Handle cases where rewards are empty or single-valued
        if len(discounted_rewards) == 0:
            print("Warning: No rewards collected in this episode.")
            continue
        elif len(discounted_rewards) == 1:
            discounted_rewards = discounted_rewards - discounted_rewards.mean()
        else:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        # Compute loss
        loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            loss.append(-log_prob * reward)  # Policy gradient loss
        loss = torch.stack(loss).sum()  # Total loss
        # Backpropagation
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Compute new gradients
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        optimizer.step()  # Update model parameters
        all_rewards.append(sum(rewards))
        #if episode==0 or episode % episode == 0:
        print(f"Episode {episode}, Total Reward: {sum(rewards)}")
    return all_rewards