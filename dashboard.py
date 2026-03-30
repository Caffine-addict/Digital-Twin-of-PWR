"""Streamlit dashboard for the PWR cooling loop digital twin.

Constraints:
- Simulation runs on button click
- Stores results in memory
- No infinite loops / no while True
"""

from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd
import altair as alt

import config as cfg
from maintenance import compute_health_metrics
from predict_future import predict_future

# Optional extensions (loaded unconditionally; they are lightweight)
from data_sources import REAL_DATA_DEFAULT_PATH, load_real_csv
from predict_next import predict_next
import lstm_predictor
from simulate_reactor import ParamsReactor, simulate_reactor
from threshold_agent import get_thresholds, load as load_agent, save as save_agent, select_action, update as update_agent
from threshold_metrics import compute_reward
from model_benchmark import benchmark as benchmark_models
from consistency_check import check_consistency
from state_estimator import KalmanConfig, kalman_filter
from uncertainty import UncertaintyConfig, compute_rolling_uncertainty
from explain import explain_dataframe

# Masters-level evaluation/experiments (optional)
from experiments import run_all as run_experiments
from ablation import run_ablation
from evaluation_metrics import compute_all_metrics
from error_analysis import analyze_errors
from calibration import analyze_score_distribution
from results_logger import save_json

from simulate_system import Params, generate_inputs, simulate
from train_model import train_and_save
from predict import load_model, score_and_flag


MODEL_PATH_DEFAULT = "models/isolation_forest.joblib"
REPORTS_DIR_DEFAULT = "runs"


_STATUS_SEVERITY = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}


@st.cache_resource
def _load_or_train_model(model_path: str) -> dict:
    path = Path(model_path)
    if not path.exists():
        train_and_save(model_path=model_path, n_steps=2000, params=Params())
    return load_model(model_path)


def _now_utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _worst_status(statuses: List[str]) -> str:
    if not statuses:
        return "NORMAL"
    return max(statuses, key=lambda s: _STATUS_SEVERITY.get(str(s), 0))


def _schedule_from_editor(df: pd.DataFrame) -> Optional[List[Dict[str, Any]]]:
    if df is None or df.empty:
        return None

    allowed = {"pump_failure", "overheating", "pressure_spike"}
    schedule: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        ftype = str(row.get("fault_type", "")).strip()
        if ftype == "" or ftype.lower() == "none":
            continue
        if ftype not in allowed:
            continue

        try:
            start_time = float(row.get("start_time"))
            end_time = float(row.get("end_time"))
            magnitude = float(row.get("magnitude"))
        except Exception:
            continue

        if end_time < start_time:
            continue

        # Domain clamp. pump_failure magnitude is [0,1] but we use [0,1] for all faults.
        if magnitude < 0.0:
            magnitude = 0.0
        if magnitude > 1.0:
            magnitude = 1.0

        schedule.append(
            {
                "fault_type": ftype,
                "start_time": start_time,
                "end_time": end_time,
                "magnitude": magnitude,
            }
        )

    return schedule or None


def _faults_df(schedule: Optional[List[Dict[str, Any]]]) -> pd.DataFrame:
    if not schedule:
        return pd.DataFrame(columns=["fault_type", "start_time", "end_time", "magnitude"])
    return pd.DataFrame(schedule)


def _signal_chart(
    *,
    df: pd.DataFrame,
    y_col: str,
    title: str,
    faults: pd.DataFrame,
) -> alt.Chart:
    base = alt.Chart(df).encode(
        x=alt.X("time:Q", title="time"),
        y=alt.Y(f"{y_col}:Q", title=title),
        tooltip=[
            alt.Tooltip("time:Q"),
            alt.Tooltip(f"{y_col}:Q"),
            alt.Tooltip("status:N"),
            alt.Tooltip("anomaly_score:Q"),
        ],
    )

    line = base.mark_line().encode(
        color=alt.Color(
            "status:N",
            scale=alt.Scale(domain=["NORMAL", "WARNING", "CRITICAL"], range=["#2c7a7b", "#b7791f", "#c53030"]),
            legend=None,
        )
    )
    points = base.mark_point(size=14, opacity=0.55).encode(
        color=alt.Color(
            "status:N",
            scale=alt.Scale(domain=["NORMAL", "WARNING", "CRITICAL"], range=["#2c7a7b", "#b7791f", "#c53030"]),
            legend=None,
        )
    )

    if faults is not None and not faults.empty:
        shade = (
            alt.Chart(faults)
            .mark_rect(opacity=0.10, color="#b7791f")
            .encode(x="start_time:Q", x2="end_time:Q")
        )
        chart = shade + line + points
    else:
        chart = line + points

    return chart.properties(height=240, title=title).interactive()


def _status_strip(df: pd.DataFrame) -> alt.Chart:
    strip = (
        alt.Chart(df)
        .mark_tick(thickness=6)
        .encode(
            x=alt.X("time:Q", title=""),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(domain=["NORMAL", "WARNING", "CRITICAL"], range=["#2c7a7b", "#b7791f", "#c53030"]),
                legend=None,
            ),
            tooltip=[alt.Tooltip("time:Q"), alt.Tooltip("status:N"), alt.Tooltip("anomaly_score:Q")],
        )
        .properties(height=40)
    )
    return strip


@st.dialog("Training Report")
def _show_training_report(report: Dict[str, Any]) -> None:
    st.json(report)
    st.download_button(
        "Download training report (JSON)",
        data=(json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        file_name="training_report.json",
        mime="application/json",
    )


@st.dialog("Predictive Risk Report")
def _show_predictive_report(report: Dict[str, Any]) -> None:
    st.json(report)
    st.download_button(
        "Download predictive report (JSON)",
        data=(json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        file_name="predictive_report.json",
        mime="application/json",
    )
def main() -> None:
    st.set_page_config(page_title="PWR Cooling Loop Digital Twin", layout="wide")
    st.title("PWR Cooling Loop Digital Twin")

    params = Params()

    if "fault_editor" not in st.session_state:
        st.session_state["fault_editor"] = pd.DataFrame(
            columns=["fault_type", "start_time", "end_time", "magnitude"]
        )

    with st.sidebar:
        st.header("Controls")
        with st.form("controls"):
            st.subheader("Simulation")
            n_steps = st.number_input("Steps", min_value=100, max_value=5000, value=800, step=100)
            heat_input = st.slider(
                "Base heat_input",
                min_value=float(params.Q_MIN),
                max_value=float(params.Q_MAX),
                value=120.0,
            )
            ambient_temp = st.slider("ambient_temp", min_value=200.0, max_value=330.0, value=290.0)
            pump_status = st.selectbox("pump_status", options=[0, 1], index=1)

            st.subheader("Fault Schedule")
            st.caption("Edit rows to add multiple faults. Empty table = no faults.")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.form_submit_button("Preset: Pump", use_container_width=True):
                    st.session_state["fault_editor"] = pd.DataFrame(
                        [
                            {
                                "fault_type": "pump_failure",
                                "start_time": 200,
                                "end_time": 400,
                                "magnitude": 1.0,
                            }
                        ]
                    )
            with c2:
                if st.form_submit_button("Preset: Heat", use_container_width=True):
                    st.session_state["fault_editor"] = pd.DataFrame(
                        [
                            {
                                "fault_type": "overheating",
                                "start_time": 200,
                                "end_time": 400,
                                "magnitude": 1.0,
                            }
                        ]
                    )
            with c3:
                if st.form_submit_button("Preset: Pressure", use_container_width=True):
                    st.session_state["fault_editor"] = pd.DataFrame(
                        [
                            {
                                "fault_type": "pressure_spike",
                                "start_time": 200,
                                "end_time": 400,
                                "magnitude": 1.0,
                            }
                        ]
                    )
            with c4:
                if st.form_submit_button("Clear", use_container_width=True):
                    st.session_state["fault_editor"] = pd.DataFrame(
                        columns=["fault_type", "start_time", "end_time", "magnitude"]
                    )

            st.session_state["fault_editor"] = st.data_editor(
                st.session_state["fault_editor"],
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "fault_type": st.column_config.SelectboxColumn(
                        "fault_type",
                        options=["pump_failure", "overheating", "pressure_spike"],
                        required=False,
                    )
                },
            )

            st.subheader("Model")
            model_path = st.text_input("model path", value=MODEL_PATH_DEFAULT)

            st.subheader("Predictive")
            horizon = st.number_input("forecast horizon (steps)", min_value=10, max_value=2000, value=100, step=10)
            persist_faults = st.checkbox("assume faults persist into forecast", value=False)

            # Optional upgrades (toggle-based)
            show_future_predictions = st.checkbox("show_future_predictions", value=False)
            show_maintenance_metrics = st.checkbox("show_maintenance_metrics", value=False)

            st.subheader("Optional Extensions")
            data_source = st.selectbox("data_source", options=["simulated", "real_csv"], index=0)
            use_full_reactor_model = st.checkbox("use_full_reactor_model", value=False)

            predictor_method = st.selectbox("predictor_method", options=["linear", "lstm"], index=0)
            train_lstm_btn = False
            if predictor_method == "lstm":
                train_lstm_btn = st.form_submit_button("Train LSTM predictor", use_container_width=True)

            adaptive_thresholding = st.checkbox("adaptive_thresholding (bandit)", value=False)

            enable_state_estimator = st.checkbox("enable_state_estimator (Kalman)", value=False)
            show_uncertainty = st.checkbox("show_uncertainty", value=False)
            show_explanations = st.checkbox("show_explanations", value=False)
            run_benchmark = st.form_submit_button("Run model benchmark", use_container_width=True)
            show_consistency = st.checkbox("show_consistency_check", value=False)

            st.subheader("Evaluation / Experiments")
            run_experiments_btn = st.form_submit_button("Run predefined experiments", use_container_width=True)
            run_ablation_btn = st.form_submit_button("Run ablation (thresholds)", use_container_width=True)
            analyze_calibration_btn = st.form_submit_button("Analyze calibration", use_container_width=True)
            analyze_errors_btn = st.form_submit_button("Run error analysis", use_container_width=True)

            load_real = False
            if data_source == "real_csv":
                load_real = st.form_submit_button("Load real data (CSV)", use_container_width=True)
            reports_dir = st.text_input("reports dir", value=REPORTS_DIR_DEFAULT)

            run_sim = st.form_submit_button("Run simulation", type="primary", use_container_width=True)
            run_pred = st.form_submit_button("Run predictive analysis", use_container_width=True)
            retrain = st.form_submit_button("Retrain model (normal only)", use_container_width=True)

    # Handle retrain with report.
    if retrain:
        saved = train_and_save(model_path=model_path, n_steps=2000, params=params)
        st.cache_resource.clear()
        report_path = Path(saved).with_suffix(".report.json")
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        st.session_state["last_training_report"] = report
        _show_training_report(report)

    schedule = _schedule_from_editor(st.session_state.get("fault_editor"))

    # Optional: train LSTM predictor (runs only on explicit trigger)
    if "lstm_state" not in st.session_state:
        st.session_state["lstm_state"] = None

    if 'train_lstm_btn' in locals() and train_lstm_btn:
        if not lstm_predictor.torch_available():
            st.error("torch not available; cannot train LSTM predictor")
        else:
            base_df = st.session_state.get("results")
            if base_df is None or len(base_df) < 50:
                st.info("Need an existing run with at least 50 points to train LSTM")
            else:
                series = base_df[["temperature", "pressure", "flow_rate"]].to_numpy(dtype=float)
                st.session_state["lstm_state"] = lstm_predictor.train_lstm(
                    series,
                    window=10,
                    epochs=50,
                    lr=1e-2,
                    hidden_size=32,
                    seed=int(cfg.SEED),
                )
                st.success("Trained LSTM predictor")

    # Optional: load real data and score (runs only on explicit trigger)
    if 'load_real' in locals() and load_real:
        try:
            df_real = load_real_csv(REAL_DATA_DEFAULT_PATH, heat_input=100.0, ambient_temp=300.0, pump_status=1)
        except Exception as e:
            st.error(f"Failed to load real CSV: {e}")
            st.stop()

        thresholds_override = None
        if adaptive_thresholding:
            agent_path = Path(reports_dir) / "threshold_agent.json"
            agent = load_agent(agent_path)
            action = select_action(agent)
            pair = get_thresholds(action)
            thresholds_override = {"warning": float(pair.warning), "critical": float(pair.critical)}

        model_bundle = _load_or_train_model(model_path)
        df_scored = score_and_flag(df_real, model_bundle, thresholds=thresholds_override)
        st.session_state["results"] = df_scored

        if adaptive_thresholding:
            m = compute_reward(df_scored)
            update_agent(agent, action, float(m["reward"]))
            save_agent(agent, agent_path)
            st.session_state["threshold_agent_metrics"] = m

    # Run simulation.
    if run_sim:
        thresholds_override = None
        agent = None
        action = None
        agent_path = None
        if adaptive_thresholding:
            agent_path = Path(reports_dir) / "threshold_agent.json"
            agent = load_agent(agent_path)
            action = select_action(agent)
            pair = get_thresholds(action)
            thresholds_override = {"warning": float(pair.warning), "critical": float(pair.critical)}

        inputs = generate_inputs(
            n_steps=int(n_steps),
            heat_input=float(heat_input),
            pump_status=int(pump_status),
            ambient_temp=float(ambient_temp),
            dt=float(params.DT),
            profile="normal",
        )

        if use_full_reactor_model:
            df = simulate_reactor(n_steps=int(n_steps), inputs=inputs, fault_schedule=schedule, params=ParamsReactor())
        else:
            df = simulate(n_steps=int(n_steps), inputs=inputs, fault_schedule=schedule, params=params)
        model_bundle = _load_or_train_model(model_path)
        df_scored = score_and_flag(df, model_bundle, thresholds=thresholds_override)
        st.session_state["results"] = df_scored
        st.session_state["last_run_meta"] = {
            "n_steps": int(n_steps),
            "heat_input": float(heat_input),
            "ambient_temp": float(ambient_temp),
            "pump_status": int(pump_status),
            "model_path": str(model_path),
            "schedule": schedule,
        }

        if adaptive_thresholding and agent is not None and action is not None and agent_path is not None:
            m = compute_reward(df_scored)
            update_agent(agent, action, float(m["reward"]))
            save_agent(agent, agent_path)
            st.session_state["threshold_agent_metrics"] = m

    # Run predictive analysis.
    if run_pred:
        # Ensure we have a base run to anchor the forecast.
        df_scored = st.session_state.get("results")
        if df_scored is None:
            inputs = generate_inputs(
                n_steps=int(n_steps),
                heat_input=float(heat_input),
                pump_status=int(pump_status),
                ambient_temp=float(ambient_temp),
                dt=float(params.DT),
                profile="normal",
            )
            df = simulate(n_steps=int(n_steps), inputs=inputs, fault_schedule=schedule, params=params)
            model_bundle = _load_or_train_model(model_path)
            df_scored = score_and_flag(df, model_bundle)
            st.session_state["results"] = df_scored

        model_bundle = _load_or_train_model(model_path)

        last = st.session_state["results"].iloc[-1]
        last_time = float(last["time"])
        initial_temperature = float(last.get("temperature_true", last["temperature"]))

        forecast_inputs = generate_inputs(
            n_steps=int(horizon),
            heat_input=float(heat_input),
            pump_status=int(pump_status),
            ambient_temp=float(ambient_temp),
            start_time=last_time + float(params.DT),
            dt=float(params.DT),
            profile="normal",
        )

        forecast_schedule = None
        if persist_faults and schedule:
            start = float(forecast_inputs[0]["time"])
            end = float(forecast_inputs[-1]["time"])
            forecast_schedule = [
                {
                    "fault_type": f["fault_type"],
                    "start_time": start,
                    "end_time": end,
                    "magnitude": float(f["magnitude"]),
                }
                for f in schedule
            ]

        if use_full_reactor_model:
            df_forecast = simulate_reactor(
                n_steps=int(horizon),
                inputs=forecast_inputs,
                fault_schedule=forecast_schedule,
                params=ParamsReactor(SEED=int(params.SEED) + 1),
                seed=int(params.SEED) + 1,
                initial_temperature=initial_temperature,
            )
        else:
            df_forecast = simulate(
                n_steps=int(horizon),
                inputs=forecast_inputs,
                fault_schedule=forecast_schedule,
                params=params,
                seed=int(params.SEED) + 1,
                initial_temperature=initial_temperature,
            )
        df_forecast_scored = score_and_flag(df_forecast, model_bundle)
        st.session_state["forecast"] = df_forecast_scored

        scores = df_forecast_scored["anomaly_score"].to_numpy(dtype=float)
        statuses = df_forecast_scored["status"].tolist()

        def _first_time(cond):
            idx = df_forecast_scored.index[cond]
            if len(idx) == 0:
                return None
            return float(df_forecast_scored.loc[idx[0], "time"])

        report = {
            "generated_at_utc": _now_utc_id(),
            "horizon_steps": int(horizon),
            "inputs": {
                "heat_input": float(heat_input),
                "ambient_temp": float(ambient_temp),
                "pump_status": int(pump_status),
                "profile": "normal",
                "persist_faults": bool(persist_faults),
            },
            "risk": {
                "worst_status": _worst_status(statuses),
                "warning_rate": float((scores < float(cfg.WARNING_SCORE_THRESHOLD)).mean()),
                "critical_rate": float((scores < float(cfg.CRITICAL_SCORE_THRESHOLD)).mean()),
                "first_time_warning": _first_time(df_forecast_scored["anomaly_score"] < float(cfg.WARNING_SCORE_THRESHOLD)),
                "first_time_critical": _first_time(df_forecast_scored["anomaly_score"] < float(cfg.CRITICAL_SCORE_THRESHOLD)),
            },
            "lowest_score_points": (
                df_forecast_scored.nsmallest(10, "anomaly_score")[
                    ["time", "temperature", "pressure", "flow_rate", "anomaly_score", "status"]
                ]
                .to_dict(orient="records")
            ),
        }

        st.session_state["last_predictive_report"] = report

        reports_path = Path(reports_dir)
        reports_path.mkdir(parents=True, exist_ok=True)
        (reports_path / f"predictive_report_{_now_utc_id()}.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _show_predictive_report(report)

    df_scored = st.session_state.get("results")
    if df_scored is None:
        st.info("Click 'Run simulation' to generate data.")
        return

    latest = df_scored.iloc[-1]
    last10_status = _worst_status(df_scored.tail(10)["status"].astype(str).tolist())
    overall_status = _worst_status(df_scored["status"].astype(str).tolist())

    s1, s2 = st.columns(2)
    with s1:
        if last10_status == "CRITICAL":
            st.error(f"Current status (worst of last 10): {last10_status}")
        elif last10_status == "WARNING":
            st.warning(f"Current status (worst of last 10): {last10_status}")
        else:
            st.success(f"Current status (worst of last 10): {last10_status}")

    with s2:
        if overall_status == "CRITICAL":
            st.error(f"Run status (worst overall): {overall_status}")
        elif overall_status == "WARNING":
            st.warning(f"Run status (worst overall): {overall_status}")
        else:
            st.success(f"Run status (worst overall): {overall_status}")

    tabs = st.tabs(["Live Run", "Anomalies", "Predictive", "Reports"])

    faults = _faults_df(schedule)

    with tabs[0]:
        st.subheader("Live Run")
        st.altair_chart(_status_strip(df_scored), use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("temperature", f"{float(latest['temperature']):.2f}")
        with m2:
            st.metric("pressure", f"{float(latest['pressure']):.2f}")
        with m3:
            st.metric("flow_rate", f"{float(latest['flow_rate']):.2f}")
        with m4:
            st.metric("anomaly_score", f"{float(latest['anomaly_score']):.3f}")

        # Optional: maintenance / trend analysis
        if show_maintenance_metrics:
            st.subheader("Maintenance / Trend Metrics (Optional)")
            if len(df_scored) >= 10:
                hm = compute_health_metrics(df_scored)
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("maintenance_risk_score", f"{float(hm['risk_score']):.3f}")
                with c2:
                    st.metric("maintenance_flag", str(bool(hm["maintenance_flag"])))
            else:
                st.info("Need at least 10 data points to compute maintenance metrics.")

        # Optional: quick linear future prediction
        if show_future_predictions:
            st.subheader("Quick Future Prediction (Linear Regression, Optional)")
            if len(df_scored) >= 10:
                window = df_scored.tail(10)
                pred = predict_next(
                    window,
                    method=str(predictor_method),
                    lstm_state=st.session_state.get("lstm_state"),
                    horizon=10,
                )
                last_time = float(df_scored.iloc[-1]["time"])
                times = [float(last_time + i) for i in range(1, 11)]
                df_pred = pd.DataFrame(
                    {
                        "time": times,
                        "T_future": pred["T_future"],
                        "P_future": pred["P_future"],
                        "F_future": pred["F_future"],
                    }
                )
                chart = (
                    alt.Chart(df_pred)
                    .transform_fold(["T_future", "P_future", "F_future"], as_=["signal", "value"])
                    .mark_line()
                    .encode(
                        x=alt.X("time:Q", title="time"),
                        y=alt.Y("value:Q", title="predicted"),
                        color=alt.Color("signal:N", legend=alt.Legend(title="signal")),
                        tooltip=[alt.Tooltip("time:Q"), alt.Tooltip("signal:N"), alt.Tooltip("value:Q")],
                    )
                    .properties(height=220)
                    .interactive()
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Need at least 10 data points to run future prediction.")

        # Optional: state estimator
        df_display = df_scored
        if enable_state_estimator:
            try:
                kcfg = KalmanConfig(q_diag=(0.05, 0.05, 0.05), r_diag=(float(cfg.SIGMA), float(cfg.SIGMA), float(cfg.SIGMA)))
                df_display = kalman_filter(df_display, config=kcfg)
            except Exception as e:
                st.warning(f"State estimator failed: {e}")

        # Optional: uncertainty
        if show_uncertainty:
            try:
                ucfg = UncertaintyConfig(window=10, z=1.96)
                df_display = compute_rolling_uncertainty(df_display, cols=["temperature", "pressure", "flow_rate"], config=ucfg)
            except Exception as e:
                st.warning(f"Uncertainty computation failed: {e}")

        # Optional: consistency check
        if show_consistency:
            try:
                df_c, summary = check_consistency(df_scored)
                st.subheader("Consistency Check (Optional)")
                st.write(summary)
            except Exception as e:
                st.warning(f"Consistency check failed: {e}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.altair_chart(
                _signal_chart(df=df_scored, y_col="temperature", title="Temperature", faults=faults),
                use_container_width=True,
            )
        with c2:
            st.altair_chart(
                _signal_chart(df=df_scored, y_col="pressure", title="Pressure", faults=faults),
                use_container_width=True,
            )
        with c3:
            st.altair_chart(
                _signal_chart(df=df_scored, y_col="flow_rate", title="Flow Rate", faults=faults),
                use_container_width=True,
            )

        st.subheader("Recent")
        st.dataframe(df_display.tail(50), use_container_width=True)

    with tabs[1]:
        st.subheader("Anomalies")
        score_chart = (
            alt.Chart(df_scored)
            .mark_line(color="#2d3748")
            .encode(
                x=alt.X("time:Q", title="time"),
                y=alt.Y("anomaly_score:Q", title="anomaly_score"),
                tooltip=[alt.Tooltip("time:Q"), alt.Tooltip("anomaly_score:Q"), alt.Tooltip("status:N")],
            )
            .properties(height=260)
            .interactive()
        )
        thresholds = pd.DataFrame(
            {
                "y": [float(cfg.WARNING_SCORE_THRESHOLD), float(cfg.CRITICAL_SCORE_THRESHOLD)],
                "label": [f"WARNING ({cfg.WARNING_SCORE_THRESHOLD})", f"CRITICAL ({cfg.CRITICAL_SCORE_THRESHOLD})"],
                "color": ["#b7791f", "#c53030"],
            }
        )
        thr = (
            alt.Chart(thresholds)
            .mark_rule(strokeDash=[4, 3])
            .encode(y="y:Q", color=alt.Color("color:N", scale=None, legend=None))
        )
        st.altair_chart(score_chart + thr, use_container_width=True)

        st.subheader("Top anomalies")
        st.dataframe(
            df_scored.nsmallest(25, "anomaly_score")[
                ["time", "temperature", "pressure", "flow_rate", "anomaly_score", "status"]
            ],
            use_container_width=True,
        )

        if show_explanations:
            st.subheader("Explanations (Optional)")
            try:
                df_exp = explain_dataframe(df_scored)
                st.dataframe(
                    df_exp.nsmallest(25, "anomaly_score")[
                        ["time", "anomaly_score", "status", "primary_cause", "confidence", "contributing_factors"]
                    ],
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"Explanation generation failed: {e}")

    with tabs[2]:
        st.subheader("Predictive")
        df_forecast = st.session_state.get("forecast")
        if df_forecast is None:
            st.info("Click 'Run predictive analysis' to generate a forecast.")
        else:
            st.altair_chart(_status_strip(df_forecast), use_container_width=True)
            c1, c2 = st.columns([2, 1])
            with c1:
                st.altair_chart(
                    _signal_chart(df=df_forecast, y_col="temperature", title="Forecast Temperature", faults=_faults_df(None)),
                    use_container_width=True,
                )
            with c2:
                last_rep = st.session_state.get("last_predictive_report")
                if last_rep:
                    st.write(last_rep.get("risk", {}))

    with tabs[3]:
        st.subheader("Reports")
        tr = st.session_state.get("last_training_report")
        pr = st.session_state.get("last_predictive_report")

        if tr:
            st.markdown("Training report (latest in session)")
            st.json(tr)
            st.download_button(
                "Download training report (JSON)",
                data=(json.dumps(tr, indent=2, sort_keys=True) + "\n").encode("utf-8"),
                file_name="training_report.json",
                mime="application/json",
            )
        else:
            st.info("No training report in this session.")

        if pr:
            st.markdown("Predictive report (latest in session)")
            st.json(pr)
            st.download_button(
                "Download predictive report (JSON)",
                data=(json.dumps(pr, indent=2, sort_keys=True) + "\n").encode("utf-8"),
                file_name="predictive_report.json",
                mime="application/json",
            )
        else:
            st.info("No predictive report in this session.")

        if run_benchmark:
            try:
                model_bundle = _load_or_train_model(model_path)
                rep = benchmark_models(df_scored, model_bundle=model_bundle)
                st.session_state["benchmark_report"] = rep
            except Exception as e:
                st.warning(f"Benchmark failed: {e}")

        bench = st.session_state.get("benchmark_report")
        if bench:
            st.markdown("Model benchmark report (latest in session)")
            st.json(bench)

        agent_metrics = st.session_state.get("threshold_agent_metrics")
        if agent_metrics:
            st.markdown("Adaptive thresholding metrics (latest in session)")
            st.json(agent_metrics)

        # Experiments
        if 'run_experiments_btn' in locals() and run_experiments_btn:
            try:
                res = run_experiments(model_path=model_path, predictor_method=str(predictor_method), lstm_state=st.session_state.get("lstm_state"))
                st.session_state["experiments_result"] = res
                path = save_json(runs_dir=reports_dir, base_name=f"experiments_{_now_utc_id()}", payload=res)
                st.session_state["experiments_result_path"] = path
            except Exception as e:
                st.warning(f"Experiments failed: {e}")

        exp = st.session_state.get("experiments_result")
        if exp:
            st.markdown("Predefined experiments (latest)")
            st.json(exp)
            p = st.session_state.get("experiments_result_path")
            if p:
                st.caption(f"Saved: {p}")

        # Ablation
        if 'run_ablation_btn' in locals() and run_ablation_btn:
            try:
                # Compare default thresholds vs a stricter pair.
                base_thr = None
                var_thr = {"warning": -0.05, "critical": -0.10}
                res = run_ablation(model_path=model_path, base_thresholds=base_thr, variant_thresholds=var_thr)
                st.session_state["ablation_result"] = res
                path = save_json(runs_dir=reports_dir, base_name=f"ablation_{_now_utc_id()}", payload=res)
                st.session_state["ablation_result_path"] = path
            except Exception as e:
                st.warning(f"Ablation failed: {e}")

        abl = st.session_state.get("ablation_result")
        if abl:
            st.markdown("Ablation result (latest)")
            st.json(abl.get("comparison", {}))
            p = st.session_state.get("ablation_result_path")
            if p:
                st.caption(f"Saved: {p}")

        # Calibration
        if 'analyze_calibration_btn' in locals() and analyze_calibration_btn:
            try:
                rep = analyze_score_distribution(df_scored)
                st.session_state["calibration_report"] = rep
                path = save_json(runs_dir=reports_dir, base_name=f"calibration_{_now_utc_id()}", payload=rep)
                st.session_state["calibration_report_path"] = path
            except Exception as e:
                st.warning(f"Calibration analysis failed: {e}")

        cal = st.session_state.get("calibration_report")
        if cal:
            st.markdown("Calibration analysis (latest)")
            st.json(cal)
            p = st.session_state.get("calibration_report_path")
            if p:
                st.caption(f"Saved: {p}")

        # Error analysis
        if 'analyze_errors_btn' in locals() and analyze_errors_btn:
            try:
                schedule = None
                meta = st.session_state.get("last_run_meta")
                if meta:
                    schedule = meta.get("schedule")
                rep = analyze_errors(df_scored=df_scored, fault_schedule=schedule)
                st.session_state["error_analysis_report"] = rep
                path = save_json(runs_dir=reports_dir, base_name=f"error_analysis_{_now_utc_id()}", payload=rep)
                st.session_state["error_analysis_report_path"] = path
            except Exception as e:
                st.warning(f"Error analysis failed: {e}")

        ea = st.session_state.get("error_analysis_report")
        if ea:
            st.markdown("Error analysis (latest)")
            st.json({k: ea[k] for k in ea if k.endswith("_count")})
            p = st.session_state.get("error_analysis_report_path")
            if p:
                st.caption(f"Saved: {p}")


if __name__ == "__main__":
    main()
