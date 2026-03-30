"""Unified future prediction interface.

Dispatches to:
- linear regression predictor (predict_future)
- optional LSTM predictor (lstm_predictor) if available and trained
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

import lstm_predictor
from predict_future import predict_future


def predict_next(
    data_window: Any,
    *,
    method: str = "linear",
    lstm_state: Optional[lstm_predictor.LstmState] = None,
    horizon: int = 10,
) -> Dict[str, list]:
    if method != "lstm":
        return predict_future(data_window)

    if (not lstm_predictor.torch_available()) or (lstm_state is None):
        return predict_future(data_window)

    if hasattr(data_window, "tail"):
        w = data_window[["temperature", "pressure", "flow_rate"]].tail(10).to_numpy(dtype=float)
    elif isinstance(data_window, list):
        w = np.asarray([[r["temperature"], r["pressure"], r["flow_rate"]] for r in data_window[-10:]], dtype=float)
    elif isinstance(data_window, dict):
        w = np.column_stack(
            [
                np.asarray(data_window["temperature"][-10:], dtype=float),
                np.asarray(data_window["pressure"][-10:], dtype=float),
                np.asarray(data_window["flow_rate"][-10:], dtype=float),
            ]
        )
    else:
        return predict_future(data_window)

    if w.shape != (10, 3):
        return predict_future(data_window)

    return lstm_predictor.predict_lstm(w, state=lstm_state, horizon=int(horizon))
