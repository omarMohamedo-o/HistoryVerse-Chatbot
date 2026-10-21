"""Agent Platform Web Portal.

Provides an enterprise-grade AI Agent Builder and Multi-Agent Workflow interface:
- Main Platform Dashboard (Metrics, Providers, Recent Executions)
- Single-Agent Builder (Identity, Providers, Capabilities, Knowledge, Playground)
- Multi-Agent Team Builder (Sequential, Parallel, Supervisor, Conditional, Human Approval)
- Workflow Execution Monitor & Auditing
- HistoryVerse AR Real-Time Voice Agent (Streaming STT, Token Streaming, TTS, Barge-In)
- Knowledge Base Management & Document RAG
- Multimodal Vision & Historical Artifact Analysis
"""

import base64
import json
import os
import uuid
import requests
import streamlit as st

API_URL = os.getenv("AGENT_API_URL", os.getenv("HISTORYVERSE_API_URL", "http://localhost:8000")).rstrip("/")

st.set_page_config(
    page_title="AI Agent Builder & Workflow Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for modern enterprise aesthetic
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e222d;
        border: 1px solid #2d3343;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 15px;
    }
    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-active { background-color: #134e48; color: #5eead4; }
    .badge-pending { background-color: #713f12; color: #fde047; }
    .badge-failed { background-color: #7f1d1d; color: #fca5a5; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session State Initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []
if "ar_messages" not in st.session_state:
    st.session_state.ar_messages = []
if "ar_artifact" not in st.session_state:
    st.session_state.ar_artifact = "Rosetta Stone"

# Sidebar Navigation and System Status
with st.sidebar:
    st.title("🤖 Agent Platform")
    st.caption("AI Agent Builder & Multi-Agent Orchestrator")

    # Backend Connectivity Check
    backend_status = "🔴 Offline"
    try:
        health_resp = requests.get(f"{API_URL}/health", timeout=2)
        if health_resp.status_code == 200:
            backend_status = "🟢 Online"
    except Exception:
        backend_status = "🔴 Offline"

    st.markdown(f"**Backend Service:** {backend_status}")
    st.caption(f"API Target: `{API_URL}`")

    mode = st.radio(
        "Navigation Menu",
        [
            "📊 Platform Dashboard",
            "🛠️ Single-Agent Builder",
            "👥 Multi-Agent Team Builder",
            "📡 Execution Monitor & Auditing",
            "🏺 HistoryVerse AR Voice Agent",
            "📚 Knowledge Base & RAG",
            "🖼️ Multimodal Vision Analysis",
        ],
    )

    st.divider()
    if st.button("🧹 Clear Active Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.agent_messages = []
        st.session_state.ar_messages = []
        st.rerun()


# Mode 1: Platform Dashboard
if mode == "📊 Platform Dashboard":
    st.title("📊 AI Platform Dashboard")
    st.markdown("Global overview of registered agents, multi-agent teams, executions, and provider infrastructure.")

    # Fetch dynamic platform counts
    agents = []
    teams = []
    executions = []
    providers = []

    try:
        r_agents = requests.get(f"{API_URL}/v1/agents", timeout=3)
        if r_agents.status_code == 200:
            agents = r_agents.json()
    except Exception:
        pass

    try:
        r_teams = requests.get(f"{API_URL}/v1/agents/teams", timeout=3)
        if r_teams.status_code == 200:
            teams = r_teams.json()
    except Exception:
        pass

    try:
        r_exec = requests.get(f"{API_URL}/v1/agents/executions", timeout=3)
        if r_exec.status_code == 200:
            executions = r_exec.json()
    except Exception:
        pass

    try:
        r_prov = requests.get(f"{API_URL}/v1/agents/providers", timeout=3)
        if r_prov.status_code == 200:
            providers = r_prov.json()
    except Exception:
        pass

    # Top KPI Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Active Agents", len(agents))
    with c2:
        st.metric("Multi-Agent Teams", len(teams))
    with c3:
        completed_runs = sum(1 for e in executions if e.get("status") == "COMPLETED")
        st.metric("Total Workflow Runs", len(executions), delta=f"{completed_runs} Successful")
    with c4:
        pending_approvals = sum(1 for e in executions if e.get("status") == "WAITING_APPROVAL")
        st.metric("Pending Approvals", pending_approvals, delta="Action Required" if pending_approvals else None)

    st.subheader("🔌 LLM Provider Engines & Capabilities")
    if providers:
        p_cols = st.columns(len(providers))
        for idx, p in enumerate(providers):
            with p_cols[idx]:
                p_status = p.get("status")
                badge = "🟢 Ready" if p_status == "active" else "🟡 Missing Credentials"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h4>{p.get('name')}</h4>
                        <p><strong>Status:</strong> {badge}</p>
                        <p><strong>Models:</strong> {', '.join(p.get('models', [])[:2])}</p>
                        <p><small>Capabilities: {', '.join(p.get('capabilities', []))}</small></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("Provider discovery endpoint unreachable or offline.")

    st.subheader("⏱️ Recent Workflow Executions")
    if executions:
        for ex in executions[:6]:
            with st.container():
                ec1, ec2, ec3, ec4 = st.columns([2, 1, 1, 3])
                ec1.write(f"**Run ID:** `{ex.get('run_id')}` ({ex.get('strategy', 'sequential').upper()})")
                status = ex.get("status", "UNKNOWN")
                ec2.write(f"Status: `{status}`")
                ec3.write(f"Time: {ex.get('created_at', '')[:19]}")
                ec4.write(f"Query: *{ex.get('query', '')[:55]}...*")
                st.divider()
    else:
        st.info("No workflow executions recorded yet. Launch a workflow from the Multi-Agent Team Builder!")


# Mode 2: Single-Agent Builder
elif mode == "🛠️ Single-Agent Builder":
    st.title("🛠️ Single-Agent Builder")
    st.markdown("Create, customize, test, and manage autonomous domain agents without writing code.")

    tab_manage, tab_create, tab_test = st.tabs(["📋 Agent Registry", "➕ Create / Edit Agent", "🧪 Test Playground"])

    # Tab 1: Agent Registry
    with tab_manage:
        st.subheader("Registered Domain Agents")
        agents_data = []
        try:
            r = requests.get(f"{API_URL}/v1/agents", timeout=5)
            if r.status_code == 200:
                agents_data = r.json()
        except Exception as exc:
            st.error(f"Failed to fetch agents: {exc}")

        if agents_data:
            for ag in agents_data:
                with st.expander(f"🤖 {ag.get('name')} — {ag.get('role')} [{ag.get('domain', 'general').upper()}]", expanded=False):
                    c_info, c_action = st.columns([3, 1])
                    with c_info:
                        st.write(f"**Agent ID:** `{ag.get('agent_id')}`")
                        st.write(f"**Domain:** `{ag.get('domain')}`")
                        st.write(f"**Model Engine:** `{ag.get('model_used')}`")
                        st.write(f"**Provider Engine:** `{ag.get('provider', 'default')}`")
                    with c_action:
                        if ag.get("agent_id") not in ["agent-mora", "archival-researcher", "historical-critic"]:
                            if st.button("🗑️ Delete Agent", key=f"del_{ag.get('agent_id')}"):
                                try:
                                    del_res = requests.delete(f"{API_URL}/v1/agents/{ag.get('agent_id')}", timeout=5)
                                    if del_res.status_code == 200:
                                        st.success(f"Agent '{ag.get('name')}' deleted.")
                                        st.rerun()
                                    else:
                                        st.error(f"Failed to delete: {del_res.text}")
                                except Exception as e:
                                    st.error(f"Delete request failed: {e}")
        else:
            st.info("No agents discovered.")

    # Tab 2: Create / Edit Agent
    with tab_create:
        st.subheader("Configure New Autonomous Agent")
        st.caption("All settings are validated and persisted to the server.")

        col_id, col_model = st.columns(2)
        with col_id:
            agent_id = st.text_input("Unique Agent Identifier:", value="medical-diagnostic-ai")
            name = st.text_input("Persona / Display Name:", value="Dr. Sarah")
            role = st.text_input("Professional Role Title:", value="Chief Clinical Consultant")
            domain = st.text_input("Knowledge Domain:", value="medicine")

        with col_model:
            provider = st.selectbox(
                "Model Provider Engine:",
                ["gemini", "openai", "ollama", "local"],
                index=0,
            )
            model_name = st.text_input(
                "Model Identifier (Optional Override):",
                placeholder="e.g. gpt-4o, gemini-1.5-pro, or llama3",
            )
            temperature = st.slider("Sampling Temperature:", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

        system_prompt = st.text_area(
            "System Prompt Instructions & Behavioral Guardrails:",
            value="You are Dr. Sarah, a specialized clinical diagnostic assistant. Provide structured, accurate, evidence-based reasoning without speculation.",
            height=130,
        )

        st.markdown("#### Capabilities & Tools")
        cap1, cap2, cap3, cap4 = st.columns(4)
        cap1.checkbox("💬 Text Reasoning", value=True, disabled=True)
        enable_rag = cap2.checkbox("📚 Isolated Knowledge RAG", value=True)
        enable_vision = cap3.checkbox("👁️ Vision Analysis", value=False)
        enable_tts = cap4.checkbox("🔊 Speech Synthesis", value=False)

        if st.button("✨ Save & Deploy Agent"):
            payload = {
                "agent_id": agent_id,
                "name": name,
                "role": role,
                "domain": domain,
                "system_prompt": system_prompt,
                "provider": provider,
                "model_name": model_name if model_name else None,
                "temperature": temperature,
            }
            with st.spinner("Deploying agent to platform orchestrator..."):
                try:
                    resp = requests.post(f"{API_URL}/v1/agents/custom", json=payload, timeout=10)
                    if resp.status_code == 200:
                        st.success(f"Agent '{name}' successfully created and persisted!")
                        st.rerun()
                    else:
                        st.error(f"Deployment failed: {resp.text}")
                except Exception as ex:
                    st.error(f"Network error: {ex}")

    # Tab 3: Test Playground
    with tab_test:
        st.subheader("Agent Conversational Playground")
        agents_list = []
        try:
            r = requests.get(f"{API_URL}/v1/agents", timeout=5)
            if r.status_code == 200:
                agents_list = r.json()
        except Exception:
            pass

        if agents_list:
            agent_map = {f"{a['name']} ({a['role']})": a["agent_id"] for a in agents_list}
            chosen_agent_label = st.selectbox("Select Target Agent:", list(agent_map.keys()))
            chosen_agent_id = agent_map[chosen_agent_label]

            # Display playground history
            for m in st.session_state.agent_messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
                    if m.get("model"):
                        st.caption(f"Model: `{m.get('model')}`")

            if test_prompt := st.chat_input("Enter message or task for the selected agent..."):
                st.session_state.agent_messages.append({"role": "user", "content": test_prompt})
                with st.chat_message("user"):
                    st.markdown(test_prompt)

                with st.chat_message("assistant"), st.spinner("Generating response..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/v1/agents/{chosen_agent_id}/run",
                            json={"query": test_prompt},
                            timeout=60,
                        )
                        if resp.status_code == 200:
                            res_json = resp.json()
                            ans = res_json.get("output", "")
                            st.markdown(ans)
                            st.caption(f"Model Engine: `{res_json.get('model_used')}` | Confidence: {res_json.get('confidence', 0.95)}")
                            st.session_state.agent_messages.append({
                                "role": "assistant",
                                "content": ans,
                                "model": res_json.get("model_used"),
                            })
                        else:
                            st.error(f"Execution failed: {resp.text}")
                    except Exception as exc:
                        st.error(f"Playground communication error: {exc}")
        else:
            st.info("No agents registered yet.")


# Mode 3: Multi-Agent Team Builder
elif mode == "👥 Multi-Agent Team Builder":
    st.title("👥 Multi-Agent Team Builder & Orchestration")
    st.markdown("Compose specialized agents into cooperative teams with verifiable execution strategies.")

    team_tab_view, team_tab_create, team_tab_run = st.tabs(["📂 Active Teams", "🏗️ Design New Team", "🚀 Run Team Workflow"])

    # Tab 1: Active Teams
    with team_tab_view:
        teams = []
        try:
            r = requests.get(f"{API_URL}/v1/agents/teams", timeout=5)
            if r.status_code == 200:
                teams = r.json()
        except Exception as e:
            st.error(f"Failed to fetch teams: {e}")

        if teams:
            for t in teams:
                with st.expander(f"👥 {t.get('name')} [{t.get('strategy', 'sequential').upper()}]", expanded=False):
                    st.write(f"**Team ID:** `{t.get('team_id')}`")
                    st.write(f"**Strategy:** `{t.get('strategy')}`")
                    st.write(f"**Member Agents:** {', '.join(t.get('members', []))}")
                    st.write(f"**Description:** {t.get('description', '')}")

                    # Render Mermaid diagram of the workflow strategy
                    strategy = t.get("strategy", "sequential").lower()
                    members = t.get("members", [])
                    if members:
                        st.markdown("##### Visual Workflow Graph:")
                        if strategy == "parallel":
                            mermaid_code = "graph LR;\n  Input --> " + " & ".join(members) + ";\n  " + " & ".join(members) + " --> Synthesizer --> Output;"
                        elif strategy == "supervisor":
                            sup = members[0] if members else "Supervisor"
                            workers = members[1:] if len(members) > 1 else members
                            mermaid_code = f"graph TD;\n  Input --> {sup};\n  {sup} --> " + f" & ".join(workers) + f";\n  " + f" & ".join(workers) + f" --> {sup} --> Output;"
                        elif strategy == "conditional":
                            router = members[0] if members else "Router"
                            targets = members[1:] if len(members) > 1 else members
                            mermaid_code = f"graph TD;\n  Input --> {router};\n  " + "\n  ".join(f"{router} -- branch --> {tgt} --> Output;" for tgt in targets)
                        else:  # sequential
                            mermaid_code = "graph LR;\n  Input --> " + " --> ".join(members) + " --> Output;"
                        st.markdown(f"```mermaid\n{mermaid_code}\n```")

                    if st.button("🗑️ Delete Team", key=f"del_team_{t.get('team_id')}"):
                        try:
                            d_resp = requests.delete(f"{API_URL}/v1/agents/teams/{t.get('team_id')}", timeout=5)
                            if d_resp.status_code == 200:
                                st.success("Team deleted.")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Deletion failed: {e}")
        else:
            st.info("No teams created yet. Use the Design New Team tab to build one!")

    # Tab 2: Design New Team
    with team_tab_create:
        st.subheader("Configure Multi-Agent Team Collaboration")

        # Get existing agents to select from
        existing_agents = []
        try:
            r = requests.get(f"{API_URL}/v1/agents", timeout=5)
            if r.status_code == 200:
                existing_agents = r.json()
        except Exception:
            pass

        agent_ids = [a["agent_id"] for a in existing_agents] if existing_agents else ["archival-researcher", "agent-mora", "historical-critic"]

        c_t1, c_t2 = st.columns(2)
        with c_t1:
            team_id = st.text_input("Team Identifier:", value="research-and-review-council")
            team_name = st.text_input("Team Display Name:", value="Archival Research & Fact Verification Council")
            team_desc = st.text_input("Description:", value="Collaborative council performing evidence retrieval and peer review.")

        with c_t2:
            team_strategy = st.selectbox(
                "Execution Orchestration Strategy:",
                ["sequential", "parallel", "supervisor", "conditional"],
                index=0,
                help="Sequential: Chained pipeline. Parallel: Concurrent execution with synthesis. Supervisor: Central coordinator. Conditional: Dynamic intent routing.",
            )
            selected_members = st.multiselect(
                "Select Member Agents (In Execution Order):",
                options=agent_ids,
                default=agent_ids[:3] if len(agent_ids) >= 3 else agent_ids,
            )

        st.markdown("#### Visual Graph Preview")
        if selected_members:
            if team_strategy == "parallel":
                m_code = "graph LR;\n  UserQuery --> " + " & ".join(selected_members) + ";\n  " + " & ".join(selected_members) + " --> Synthesis --> FinalResponse;"
            elif team_strategy == "supervisor":
                sup = selected_members[0]
                workers = selected_members[1:] if len(selected_members) > 1 else selected_members
                m_code = f"graph TD;\n  UserQuery --> {sup};\n  {sup} --> " + f" & ".join(workers) + f";\n  " + f" & ".join(workers) + f" --> {sup} --> FinalResponse;"
            elif team_strategy == "conditional":
                router = selected_members[0]
                branches = selected_members[1:] if len(selected_members) > 1 else selected_members
                m_code = f"graph TD;\n  UserQuery --> {router};\n  " + "\n  ".join(f"{router} -- Category --> {b} --> FinalResponse;" for b in branches)
            else:
                m_code = "graph LR;\n  UserQuery --> " + " --> ".join(selected_members) + " --> FinalResponse;"
            st.markdown(f"```mermaid\n{m_code}\n```")

        if st.button("✨ Save and Register Team"):
            if not selected_members:
                st.warning("Please select at least one member agent.")
            else:
                t_payload = {
                    "team_id": team_id,
                    "name": team_name,
                    "strategy": team_strategy,
                    "members": selected_members,
                    "description": team_desc,
                    "config": {
                        "aggregator_id": selected_members[0],
                        "supervisor_id": selected_members[0],
                        "router_id": selected_members[0],
                    },
                }
                try:
                    save_res = requests.post(f"{API_URL}/v1/agents/teams", json=t_payload, timeout=10)
                    if save_res.status_code == 200:
                        st.success(f"Team '{team_name}' successfully created!")
                        st.rerun()
                    else:
                        st.error(f"Failed to save team: {save_res.text}")
                except Exception as ex:
                    st.error(f"Network error: {ex}")

    # Tab 3: Run Team Workflow
    with team_tab_run:
        st.subheader("Execute Multi-Agent Workflow")
        teams_available = []
        try:
            r = requests.get(f"{API_URL}/v1/agents/teams", timeout=5)
            if r.status_code == 200:
                teams_available = r.json()
        except Exception:
            pass

        if teams_available:
            t_options = {f"{t['name']} ({t['strategy'].upper()})": t["team_id"] for t in teams_available}
            chosen_team_label = st.selectbox("Select Team to Execute:", list(t_options.keys()))
            chosen_team_id = t_options[chosen_team_label]

            workflow_query = st.text_area("Task Query / Mandate for the Team:", value="Examine the historical evidence of the Greco-Roman museum founding and verify its key claims.", height=90)
            enable_human_checkpoint = st.checkbox("🛑 Require Human-in-the-Loop Approval Checkpoint before proceeding", value=False)

            if st.button("🚀 Run Workflow Execution") and workflow_query:
                with st.spinner("Executing multi-agent workflow across team pipeline..."):
                    try:
                        exec_payload = {
                            "query": workflow_query,
                            "require_human_approval": enable_human_checkpoint,
                        }
                        run_res = requests.post(
                            f"{API_URL}/v1/agents/teams/{chosen_team_id}/run",
                            json=exec_payload,
                            timeout=90,
                        )
                        if run_res.status_code == 200:
                            run_data = run_res.json()
                            if run_data.get("status") == "WAITING_APPROVAL":
                                st.warning(f"⏸️ **Workflow Paused at Human Checkpoint!** Run ID: `{run_data.get('run_id')}`")
                                st.info("Navigate to the **Execution Monitor & Auditing** tab to review and approve/reject.")
                            else:
                                st.success(f"✅ Workflow Execution Completed in {run_data.get('duration_seconds', 0)}s!")
                                st.markdown("### Final Synthesized Resolution:")
                                st.markdown(run_data.get("final_answer", ""))

                                st.markdown("#### Inter-Agent Execution Trace:")
                                for step in run_data.get("steps", []):
                                    with st.expander(f"🔹 {step.get('step_id')}: {step.get('agent_name')} ({step.get('role')})"):
                                        st.markdown(step.get("output", ""))
                                        st.caption(f"Model Engine: `{step.get('model_used')}`")
                        else:
                            st.error(f"Execution failed: {run_res.text}")
                    except Exception as exc:
                        st.error(f"Workflow execution error: {exc}")
        else:
            st.info("No teams available. Create one in the Design New Team tab first!")


# Mode 4: Execution Monitor & Auditing
elif mode == "📡 Execution Monitor & Auditing":
    st.title("📡 Execution Monitor & Auditing")
    st.markdown("Real-time audit log of multi-agent workflow runs, intermediate step artifacts, and human approval checkpoints.")

    executions = []
    try:
        r = requests.get(f"{API_URL}/v1/agents/executions", timeout=5)
        if r.status_code == 200:
            executions = r.json()
    except Exception as e:
        st.error(f"Failed to fetch execution records: {e}")

    if executions:
        for ex in executions:
            run_id = ex.get("run_id")
            status = ex.get("status", "UNKNOWN")
            is_pending = (status == "WAITING_APPROVAL")

            badge_class = "badge-pending" if is_pending else ("badge-active" if status == "COMPLETED" else "badge-failed")

            with st.expander(f"Run `{run_id}` | Strategy: {ex.get('strategy', 'sequential').upper()} | Status: {status}", expanded=is_pending):
                c_detail, c_ctrl = st.columns([3, 1])
                with c_detail:
                    st.write(f"**Task Query:** *{ex.get('query')}*")
                    st.write(f"**Execution Timestamp:** {ex.get('created_at')}")
                    if ex.get("final_answer"):
                        st.markdown("**Final Answer:**")
                        st.markdown(ex.get("final_answer"))

                    # Step trace
                    steps = ex.get("steps", [])
                    if steps:
                        st.markdown("**Step Outputs:**")
                        for s in steps:
                            st.write(f"• **{s.get('agent_name', s.get('agent_id'))}** ({s.get('role', 'Specialist')}): {s.get('output', '')[:120]}...")

                with c_ctrl:
                    if is_pending:
                        st.warning("⚠️ **Human Approval Required**")
                        user_notes = st.text_input("Reviewer Notes:", key=f"notes_{run_id}")
                        col_app, col_rej = st.columns(2)
                        with col_app:
                            if st.button("✅ Approve", key=f"app_{run_id}"):
                                try:
                                    app_resp = requests.post(
                                        f"{API_URL}/v1/agents/executions/{run_id}/approve",
                                        json={"approved": True, "comment": user_notes},
                                        timeout=60,
                                    )
                                    if app_resp.status_code == 200:
                                        st.success("Run approved and completed!")
                                        st.rerun()
                                    else:
                                        st.error(f"Approval failed: {app_resp.text}")
                                except Exception as exc:
                                    st.error(f"Error: {exc}")
                        with col_rej:
                            if st.button("❌ Reject", key=f"rej_{run_id}"):
                                try:
                                    rej_resp = requests.post(
                                        f"{API_URL}/v1/agents/executions/{run_id}/approve",
                                        json={"approved": False, "comment": user_notes},
                                        timeout=10,
                                    )
                                    if rej_resp.status_code == 200:
                                        st.warning("Run rejected.")
                                        st.rerun()
                                except Exception as exc:
                                    st.error(f"Error: {exc}")
    else:
        st.info("No workflow executions logged yet.")


# Mode 5: HistoryVerse AR Voice Agent (Real-Time Voice, Turn-Taking, Barge-In)
elif mode == "🏺 HistoryVerse AR Voice Agent":
    st.title("🏺 HistoryVerse AR Real-Time Voice Agent")
    st.markdown("Live conversational voice experience anchored to detected ancient statues and monuments.")

    ar_col1, ar_col2 = st.columns([3, 1])
    with ar_col1:
        statue_selection = st.selectbox(
            "Detected Monument / Statue Artifact:",
            [
                "Rosetta Stone",
                "Golden Mask of Tutankhamun",
                "Pharos Lighthouse of Alexandria",
                "Statue of Serapis",
                "Bust of Nefertiti",
                "Great Sphinx of Giza",
            ],
            index=0,
        )
        if statue_selection != st.session_state.ar_artifact:
            st.session_state.ar_artifact = statue_selection
            st.info(f"AR Camera re-anchored to **{statue_selection}**!")

    with ar_col2:
        st.markdown("**Voice Capabilities:**")
        st.caption("✅ Streaming STT (Whisper)")
        st.caption("✅ Token-level Streaming LLM")
        st.caption("✅ Sentence-chunked Audio (TTS)")
        st.caption("✅ Live Barge-in Interruption")

    st.subheader(f"Conversational Guide: {st.session_state.ar_artifact}")

    for msg in st.session_state.ar_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("audio_url"):
                st.audio(msg["audio_url"])

    col_input, col_barge = st.columns([4, 1])
    with col_input:
        ar_query = st.chat_input(f"Speak or ask Mora about {st.session_state.ar_artifact}...")
    with col_barge:
        if st.button("🛑 Barge-in (Interrupt)"):
            st.warning("Interruption signal transmitted! Guide stopped speaking.")

    if ar_query:
        st.session_state.ar_messages.append({"role": "user", "content": ar_query})
        with st.chat_message("user"):
            st.markdown(ar_query)

        with st.chat_message("assistant"), st.spinner(f"Mora is inspecting {st.session_state.ar_artifact}..."):
            try:
                chat_payload = {
                    "artifact_name": st.session_state.ar_artifact,
                    "query": ar_query,
                    "session_id": st.session_state.session_id,
                    "speak_response": True,
                }
                resp = requests.post(f"{API_URL}/v1/ar/chat", json=chat_payload, timeout=60)
                if resp.status_code == 200:
                    ans_data = resp.json()
                    answer_text = ans_data.get("answer", "")
                    audio_path = ans_data.get("audio_url")
                    st.markdown(answer_text)
                    if audio_path:
                        st.audio(audio_path)
                    st.session_state.ar_messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "audio_url": audio_path,
                    })
                else:
                    st.error(f"AR Guide error: {resp.text}")
            except Exception as exc:
                st.error(f"AR Guide communication error: {exc}")


# Mode 6: Knowledge Base & RAG
elif mode == "📚 Knowledge Base & RAG":
    st.subheader("📚 Knowledge Base Ingestion & Document Q&A")
    st.caption("Upload documents to build dedicated, tenant-isolated vector collections.")

    uploaded_file = st.file_uploader("Upload Knowledge Document (PDF, TXT, MD):", type=["pdf", "txt", "md"])
    target_agent_id = st.text_input("Bind to Specific Agent ID (Optional Isolation Scope):", value="agent-mora")

    doc_query = st.text_input("Test Knowledge Retrieval Query:")
    if st.button("🔍 Ingest & Analyze"):
        if not uploaded_file or not doc_query:
            st.warning("Please provide both a document and a test query.")
        else:
            with st.spinner("Processing document through RAG pipeline..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {"query": doc_query}
                    resp = requests.post(f"{API_URL}/v1/chat/file", files=files, data=data, timeout=90)
                    if resp.status_code == 200:
                        res = resp.json()
                        st.markdown(f"### Answer:\n{res.get('answer', '')}")
                        if res.get("sources"):
                            st.markdown("#### Retrieved Evidence Sources:")
                            for src in res["sources"]:
                                st.markdown(f"- `{src}`")
                    else:
                        st.error(f"Ingest failed: {resp.text}")
                except Exception as exc:
                    st.error(f"Request error: {exc}")


# Mode 7: Multimodal Vision Analysis
elif mode == "🖼️ Multimodal Vision Analysis":
    st.subheader("🖼️ Multimodal Vision & Historical Artifact Analysis")
    st.caption("Upload an image of a statue, artifact, or historical inscription for visual analysis.")

    uploaded_img = st.file_uploader("Upload Image:", type=["jpg", "jpeg", "png", "webp"])
    vision_prompt = st.text_input(
        "Vision Inquiry:",
        value="Identify this historical artifact and describe its period, provenance, and historical significance.",
    )

    if uploaded_img:
        st.image(uploaded_img, caption="Target Artifact", use_container_width=True)
        if st.button("🔎 Analyze Image"):
            with st.spinner("Analyzing visual artifact..."):
                try:
                    files = {"file": (uploaded_img.name, uploaded_img.getvalue(), uploaded_img.type)}
                    data = {"prompt": vision_prompt}
                    resp = requests.post(f"{API_URL}/v1/vision/analyze", files=files, data=data, timeout=60)
                    if resp.status_code == 200:
                        out = resp.json()
                        st.markdown(f"### Visual Insights:\n{out.get('analysis', out.get('answer', ''))}")
                    else:
                        st.error(f"Vision analysis failed: {resp.text}")
                except Exception as exc:
                    st.error(f"Vision request error: {exc}")
