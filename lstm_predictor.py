"""Optional LSTM predictor for T/P/F.

- Predicts T, P, F only
- Horizon = 10 (handled by caller)
- Run only on user trigger (dashboard will call train/predict explicitly)
- Optional: callers can fall back if torch not available
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


@dataclass
class LstmState:
    model_state_dict: Dict[str, Any]
    input_mean: np.ndarray
    input_std: np.ndarray
    hidden_size: int


def train_lstm(
    series: np.ndarray,
    *,
    window: int = 10,
    epochs: int = 50,
    lr: float = 1e-2,
    hidden_size: int = 32,
    seed: int = 42,
) -> LstmState:
    """Train a tiny LSTM to predict next-step (T,P,F) from past window.

    series: shape (N,3) for [T,P,F]
    """
    if not torch_available():
        raise RuntimeError("torch is not available; LSTM predictor is disabled")

    import torch
    import torch.nn as nn

    torch.manual_seed(int(seed))

    x = np.asarray(series, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("series must be shape (N,3) for [T,P,F]")
    if x.shape[0] < window + 2:
        raise ValueError("need more data to train LSTM")

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    xn = (x - mean) / std

    X_list = []
    Y_list = []
    for i in range(window, xn.shape[0]):
        X_list.append(xn[i - window : i])
        Y_list.append(xn[i])

    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    Y = torch.tensor(np.stack(Y_list), dtype=torch.float32)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=3, hidden_size=int(hidden_size), batch_first=True)
            self.fc = nn.Linear(int(hidden_size), 3)

        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.fc(last)

    model = _Model()
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(int(epochs)):
        pred = model(X)
        loss = loss_fn(pred, Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return LstmState(
        model_state_dict=model.state_dict(),
        input_mean=mean,
        input_std=std,
        hidden_size=int(hidden_size),
    )


def predict_lstm(
    window_values: np.ndarray,
    *,
    state: LstmState,
    horizon: int = 10,
) -> Dict[str, list]:
    """Predict horizon steps ahead using autoregressive rollout."""
    if not torch_available():
        raise RuntimeError("torch is not available; LSTM predictor is disabled")

    import torch
    import torch.nn as nn

    w = np.asarray(window_values, dtype=float)
    if w.shape != (10, 3):
        raise ValueError("window_values must be shape (10,3)")

    mean = np.asarray(state.input_mean, dtype=float)
    std = np.asarray(state.input_std, dtype=float)
    std = np.where(std == 0.0, 1.0, std)
    hidden_size = int(state.hidden_size)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=3, hidden_size=int(hidden_size), batch_first=True)
            self.fc = nn.Linear(int(hidden_size), 3)

        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.fc(last)

    model = _Model()
    model.load_state_dict(state.model_state_dict)
    model.eval()

    cur = (w - mean) / std
    preds = []
    with torch.no_grad():
        for _ in range(int(horizon)):
            x = torch.tensor(cur[None, :, :], dtype=torch.float32)
            yn = model(x).numpy()[0]
            y = yn * std + mean
            preds.append(y)
            cur = np.vstack([cur[1:], yn])

    preds = np.asarray(preds, dtype=float)
    return {
        "T_future": [float(v) for v in preds[:, 0].tolist()],
        "P_future": [float(v) for v in preds[:, 1].tolist()],
        "F_future": [float(v) for v in preds[:, 2].tolist()],
    }
