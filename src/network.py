"""
Neural Network Architecture for DQN
"""

import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Deep Q-Network for Celeste Classic.
    
    Architecture:
        Input (6) → Dense(256) → ReLU → Dense(256) → ReLU → Dense(128) → ReLU → Output (15)
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize the DQN.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            hidden_dim: Size of hidden layers
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Q-values for each action, shape (batch_size, action_dim)
        """
        return self.network(x)


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture (optional improvement).
    
    Separates value and advantage streams for better learning.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared feature layer
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


if __name__ == "__main__":
    # Test the networks
    state_dim = 6
    action_dim = 15
    batch_size = 32
    
    # Test DQN
    dqn = DQN(state_dim, action_dim)
    test_input = torch.randn(batch_size, state_dim)
    output = dqn(test_input)
    print(f"DQN output shape: {output.shape}")  # Should be (32, 15)
    
    # Test Dueling DQN
    dueling = DuelingDQN(state_dim, action_dim)
    output = dueling(test_input)
    print(f"Dueling DQN output shape: {output.shape}")  # Should be (32, 15)
    
    # Count parameters
    dqn_params = sum(p.numel() for p in dqn.parameters())
    dueling_params = sum(p.numel() for p in dueling.parameters())
    print(f"DQN parameters: {dqn_params:,}")
    print(f"Dueling DQN parameters: {dueling_params:,}")
    
    print("✅ Networks working!")
