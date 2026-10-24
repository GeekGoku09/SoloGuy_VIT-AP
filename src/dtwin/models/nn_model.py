
import numpy as np
import pandas as pd
import torch
from torch import nn


class GRUFusionForecaster(nn.Module):
    """
    GRU Neural Model for time-series conditioning + static EHR features.
    """
    def __init__(self, ts_input_dim: int = 5, static_dim: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.gru = nn.GRU(ts_input_dim, hidden_dim, batch_first=True)
        self.static_fc = nn.Linear(static_dim, 16)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x_ts)
        ts_feat = out[:, -1, :] # last time step
        st_feat = torch.relu(self.static_fc(x_static))
        fused = torch.cat([ts_feat, st_feat], dim=-1)
        prob = self.head(fused)
        return prob.squeeze(-1)

def train_and_eval_gru_model(X: pd.DataFrame, y: np.ndarray) -> float:
    """Trains PyTorch GRU model and returns training loss."""
    torch.manual_seed(42)
    model = GRUFusionForecaster()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    # Create dummy 3D time series tensor (batch, seq_len, ts_dim) and static tensor
    batch_size = len(X)
    x_ts = torch.randn(batch_size, 12, 5) # 12 past steps
    x_st = torch.randn(batch_size, 4)
    y_t = torch.tensor(y, dtype=torch.float32)

    model.train()
    for _ in range(10):
        optimizer.zero_grad()
        preds = model(x_ts, x_st)
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()

    return float(loss.item())
