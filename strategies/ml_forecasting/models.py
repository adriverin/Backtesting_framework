# strategies/ml_forecasting/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MLConfig

class SimpleModel(nn.Module):
    """Simplified model architecture designed to prevent overfitting."""
    def __init__(self, input_dim: int, config: MLConfig):
        super().__init__()
        self.config = config
        self.output_dim = config.n_quantiles
        
        # Simple two-layer MLP
        self.fc1 = nn.Linear(input_dim, config.hidden_sizes[0])
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc2 = nn.Linear(config.hidden_sizes[0], self.output_dim)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

def create_model(input_dim: int, config: MLConfig) -> nn.Module:
    """Factory function to create the model."""
    model = SimpleModel(input_dim, config)
    return model.to(config.device)