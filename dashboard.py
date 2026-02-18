"""
dashboard.py
────────────
Real-time UI and Audit Log Viewer for Jarvis.
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import time

st.set_page_config(page_title="Jarvis Dashboard", page_icon="🤖", layout="wide")

# ─── Config ───
LOG_DIR = Path("outputs/Jarvis-Session")

# ─── Sidebar: Session Selector ───
st.sidebar.title("🧠 Jarvis Core")
st.sidebar.markdown("---")

if not LOG_DIR.exists():
    st.error(f"No logs found in {LOG_DIR}")
    st.stop()

# Find sessions (folders inside outputs/Jarvis-Session)
# Adjust logic based on your AuditLogger structure. 
# Assuming AuditLogger creates files in outputs/Jarvis-Session/ directly 
# or makes subfolders based on session_id.
# Based on your file: AuditLogger uses `outputs/Jarvis-Session/filename.jsonl`.
# It doesn't seem to create subfolders per session in the provided code, 
# but appends to shared files. We will read the shared files and filter by Session ID.

# 1. Load Data
@st.cache_data(ttl=5)
def load_data(filename):
    path = LOG_DIR / filename
    if not path.exists(): return []
    data = []
    with open(path, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except: pass
    return data

# Load Trace and Obs
plans = load_data("agent_plan.jsonl")
traces = load_data("execution_trace.jsonl")
observations = load_data("tool_observations.jsonl")
risks = load_data("risk_assessment_log.jsonl")

# Extract Session IDs
all_sessions = sorted(list(set([d['session_id'] for d in traces])), reverse=True)

if not all_sessions:
    st.warning("Waiting for Jarvis to start a session...")
    st.stop()

selected_session = st.sidebar.selectbox("Select Session", all_sessions)

# Filter Data
session_traces = [t for t in traces if t['session_id'] == selected_session]
session_obs = [o for o in observations if o.get('session_id') == selected_session] 
# Note: Your current ToolRouter/AuditLogger might not tag observations with session_id 
# directly in the file format provided. 
# If not, we view global logs or you update AuditLogger to include session_id in all files.
# For now, we'll display what we have.

# ─── Main Dashboard ───

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Activity Feed (Audit Log)")
    
    # Combine Plans and Traces for a timeline
    timeline = []
    for t in session_traces:
        timeline.append({"time": t['timestamp'], "type": "GOAL", "data": t['data']})
        
    # Sort by time
    # timeline.sort(key=lambda x: x['time'], reverse=True)

    for item in reversed(timeline):
        data = item['data']
        with st.expander(f"Goal: {data.get('goal', 'Unknown')} ({data.get('duration_seconds', 0)}s)", expanded=True):
            st.markdown(f"**Outcome:** {'✅ Success' if data.get('success') else '❌ Failed'}")
            st.caption(f"Reason: {data.get('stop_reason')}")
            
            if data.get('plan'):
                st.info(f"**Plan:** {len(data['plan']['steps'])} steps")
            
            if data.get('final_response'):
                st.markdown(f"**Jarvis:** {data['final_response']}")

with col2:
    st.subheader("⚙️ Tool Execution")
    # Show global recent tool usage if session linking isn't perfect yet
    st.caption("Recent tool calls system-wide")
    
    df_obs = pd.DataFrame([x['data'] for x in observations])
    if not df_obs.empty:
        # Simplify view
        df_display = df_obs[['tool_name', 'execution_status', 'duration_seconds']]
        st.dataframe(df_display.tail(10), use_container_width=True)
    else:
        st.write("No tools used yet.")

    st.subheader("🛡️ Risk Assessments")
    df_risk = pd.DataFrame([x['data'] for x in risks])
    if not df_risk.empty:
        st.line_chart(df_risk['composite_score'].tail(20))
        latest = df_risk.iloc[-1]
        st.metric("Current Risk Level", latest['level'], f"{latest['composite_score']:.2f}")

# Auto-refresh
time.sleep(2)
st.rerun()