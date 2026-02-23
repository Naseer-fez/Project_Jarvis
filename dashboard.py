"""
Jarvis Dashboard (V4-ready)
Run: streamlit run dashboard.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Jarvis Control", page_icon="J", layout="wide")

TRACE_DIR = Path("outputs/Jarvis-Session")
CONTROL_FILE = Path("runtime/control_flags.json")
CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)
if not CONTROL_FILE.exists():
    CONTROL_FILE.write_text("{}", encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _write_control(payload: dict) -> None:
    CONTROL_FILE.write_text(json.dumps(payload), encoding="utf-8")


st.title("Jarvis Runtime Dashboard")
st.caption("Real-time transcription, state, action logs, and manual overrides.")

if not TRACE_DIR.exists():
    st.warning(f"Trace directory not found: {TRACE_DIR}")
    st.stop()

transcriptions = _load_jsonl(TRACE_DIR / "voice_transcription.jsonl")
states = _load_jsonl(TRACE_DIR / "state_trace.jsonl")
actions = _load_jsonl(TRACE_DIR / "action_trace.jsonl")
errors = _load_jsonl(TRACE_DIR / "error_trace.jsonl")

left, right = st.columns([2, 1])

with left:
    st.subheader("Real-Time Transcription")
    if transcriptions:
        rows = []
        for row in transcriptions[-50:]:
            data = row.get("data", {})
            rows.append(
                {
                    "timestamp": row.get("timestamp"),
                    "text": data.get("text", ""),
                    "final": data.get("is_final", False),
                    "confidence": data.get("confidence"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No transcription events yet.")

    st.subheader("Action Log")
    if actions:
        rows = []
        for row in actions[-100:]:
            data = row.get("data", {})
            rows.append(
                {
                    "timestamp": row.get("timestamp"),
                    "action": data.get("action"),
                    "status": data.get("status"),
                    "step_id": data.get("step_id"),
                    "duration_s": data.get("duration_s"),
                    "error": data.get("error"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No action traces yet.")

with right:
    st.subheader("Agent State View")
    if states:
        recent = [row.get("data", {}) for row in states[-15:]]
        df = pd.DataFrame(recent)
        st.dataframe(df, use_container_width=True, hide_index=True)
        if not df.empty and "state" in df.columns:
            st.metric("Current State", str(df.iloc[-1]["state"]))
    else:
        st.info("No state transitions yet.")

    st.subheader("Manual Override Controls")
    if st.button("Interrupt Current Task", use_container_width=True):
        _write_control({"interrupt": True, "ts": time.time()})
        st.success("Interrupt signal sent.")

    if st.button("Enable Failsafe (Disable Actions)", use_container_width=True):
        _write_control({"disable_actions": True, "ts": time.time()})
        st.success("Failsafe disable signal sent.")

    if st.button("Resume Actions", use_container_width=True):
        _write_control({"resume_actions": True, "ts": time.time()})
        st.success("Resume signal sent.")

    st.subheader("Error Trace")
    if errors:
        rows = []
        for row in errors[-20:]:
            data = row.get("data", {})
            rows.append(
                {
                    "timestamp": row.get("timestamp"),
                    "source": data.get("source"),
                    "message": data.get("message"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No errors logged.")

time.sleep(1.0)
st.rerun()
