"""Agent Streamlit UI.

Provides an interactive user interface for:
- Multi-Domain Agent Hub (Dynamic Domain Models)
- Historical Guide (Mora Persona Preset)
- AR Mode (Statue & Artifact Guided Tour)
- Chat with Domain Documents (RAG)
- Multimodal Vision & Artifact Analysis
"""

import os
import uuid

import requests
import streamlit as st

API_URL = os.getenv("AGENT_API_URL", os.getenv("HISTORYVERSE_API_URL", "http://localhost:8000")).rstrip("/")

st.set_page_config(
    page_title="HistoryVerse AI",
    page_icon="🏛️",
    layout="wide",
)

st.title("🏛️ HistoryVerse AI Portal")
st.markdown("Multimodal AI Agent Platform featuring HistoryVerse Agent & HistoryVerse AR Agent")

# Sidebar navigation and backend status
with st.sidebar:
    st.header("⚙️ Configuration")
    backend_status = "🔴 Offline"
    try:
        health_resp = requests.get(f"{API_URL}/health", timeout=3)
        if health_resp.status_code == 200:
            backend_status = "🟢 Online"
    except Exception:
        backend_status = "🔴 Offline"

    st.write(f"**Backend Status**: {backend_status}")
    st.write(f"**API URL**: `{API_URL}`")

    mode = st.radio(
        "Choose Mode:",
        [
            "🏛️ HistoryVerse Agent (Mora Assistant)",
            "🏺 HistoryVerse AR Agent (Mora AR Guide)",
            "🤖 Multi-Domain Agent Hub (Medicine, Law, Custom)",
            "📜 Chat with Documents",
            "🖼️ Historical Vision Analysis",
        ],
    )

    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.ar_active_artifact = None
        st.rerun()

# Initialize session states
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ar_active_artifact" not in st.session_state:
    st.session_state.ar_active_artifact = None


# -------------------------------------------------------------
# Mode 1: HistoryVerse Agent (Mora Assistant)
# -------------------------------------------------------------
if mode == "🏛️ HistoryVerse Agent (Mora Assistant)":
    st.subheader("🏛️ HistoryVerse Historical Dialogue")
    st.caption("Converse with Mora, your AI historical research assistant.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("audio_url"):
                st.audio(msg["audio_url"])

    speak_audio = st.checkbox("🔊 Read response aloud (TTS)", value=False)

    if user_prompt := st.chat_input(
        "Ask a question about history, civilizations, or archaeology..."
    ):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with (
            st.chat_message("assistant"),
            st.spinner("Mora is consulting the historical archives..."),
        ):
            try:
                payload = {
                    "query": user_prompt,
                    "conversation_id": st.session_state.session_id,
                    "speak_response": speak_audio,
                }
                resp = requests.post(f"{API_URL}/v1/chat", json=payload, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "")
                    audio_url = data.get("audio_url")
                    st.markdown(answer)
                    if audio_url:
                        st.audio(audio_url)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "audio_url": audio_url}
                    )
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as exc:
                st.error(f"Failed to communicate with HistoryVerse API: {exc}")


# -------------------------------------------------------------
# Mode 2: HistoryVerse AR Agent (Mora AR Guide)
# -------------------------------------------------------------
elif mode == "🏺 HistoryVerse AR Agent (Mora AR Guide)":
    st.subheader("🏺 HistoryVerse AR Mode: Guided Artifact Exploration")
    st.caption(
        "Anchor your tour guide Mora to a detected statue, monument, or historical artifact."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        artifact_input = st.text_input(
            "Detected Artifact or Statue Name:",
            value=st.session_state.ar_active_artifact or "Statue of Serapis",
        )
    with col2:
        lang_choice = st.selectbox("Guide Language:", ["en", "ar"])

    speak_ar_audio = st.checkbox("🔊 Enable audio tour guide", value=False)

    if (
        st.button("🚀 Start Guided AR Exploration")
        or st.session_state.ar_active_artifact != artifact_input
    ):
        st.session_state.ar_active_artifact = artifact_input
        with st.spinner(f"Initializing AR guide for '{artifact_input}'..."):
            try:
                start_payload = {
                    "artifact_name": artifact_input,
                    "session_id": st.session_state.session_id,
                    "language": lang_choice,
                    "speak_response": speak_ar_audio,
                }
                resp = requests.post(
                    f"{API_URL}/v1/ar/start",
                    json=start_payload,
                    timeout=60,
                )
                if resp.status_code == 200:
                    start_data = resp.json()
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": start_data.get("answer", ""),
                            "audio_url": start_data.get("audio_url"),
                        }
                    )
                else:
                    st.error(f"Failed to start AR mode: {resp.text}")
            except Exception as exc:
                st.error(f"AR initialization error: {exc}")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("audio_url"):
                st.audio(msg["audio_url"])

    if ar_prompt := st.chat_input(
        f"Ask about {st.session_state.ar_active_artifact or 'the artifact'}..."
    ):
        st.session_state.messages.append({"role": "user", "content": ar_prompt})
        with st.chat_message("user"):
            st.markdown(ar_prompt)

        with st.chat_message("assistant"), st.spinner("Guide is answering..."):
            try:
                chat_payload = {
                    "artifact_name": st.session_state.ar_active_artifact or "Artifact",
                    "query": ar_prompt,
                    "session_id": st.session_state.session_id,
                    "speak_response": speak_ar_audio,
                }
                resp = requests.post(
                    f"{API_URL}/v1/ar/chat",
                    json=chat_payload,
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    ans = data.get("answer", "")
                    st.markdown(ans)
                    if data.get("audio_url"):
                        st.audio(data.get("audio_url"))
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": ans,
                            "audio_url": data.get("audio_url"),
                        }
                    )
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as exc:
                st.error(f"AR chat error: {exc}")


# -------------------------------------------------------------
# Mode 3: Chat with Historical Documents
# -------------------------------------------------------------
elif mode == "📜 Chat with Documents":
    st.subheader("📜 Chat with Historical Documents & Manuscripts")
    st.caption("Upload a historical manuscript, research paper, or text to converse with it.")

    uploaded_file = st.file_uploader(
        "Upload PDF, TXT, or Markdown document", type=["pdf", "txt", "md"]
    )

    doc_query = st.text_input("Question about the document:")
    if st.button("🔍 Analyze and Answer"):
        if not uploaded_file or not doc_query:
            st.warning("Please upload a file and enter a question.")
        else:
            with st.spinner("Processing document through RAG pipeline..."):
                try:
                    files = {
                        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                    }
                    data = {"query": doc_query}
                    resp = requests.post(
                        f"{API_URL}/v1/chat/file", files=files, data=data, timeout=90
                    )
                    if resp.status_code == 200:
                        res = resp.json()
                        st.markdown(f"### Answer:\n{res.get('answer', '')}")
                        if res.get("sources"):
                            st.markdown("#### Citations & Sources:")
                            for src in res["sources"]:
                                st.markdown(f"- {src}")
                    else:
                        st.error(f"Document analysis failed: {resp.text}")
                except Exception as exc:
                    st.error(f"Request error: {exc}")


# -------------------------------------------------------------
# Mode 4: Historical Vision Analysis
# -------------------------------------------------------------
elif mode == "🖼️ Historical Vision Analysis":
    st.subheader("🖼️ Historical Vision & Artifact Analysis")
    st.caption(
        "Upload an image of a historical site, artifact, or painting for AI identification and analysis."
    )

    uploaded_img = st.file_uploader("Upload an Image:", type=["jpg", "jpeg", "png", "webp"])
    custom_vision_prompt = st.text_input(
        "Optional specific inquiry:",
        value="Identify this historical artifact and describe its period, provenance, and historical significance.",
    )

    if uploaded_img:
        st.image(uploaded_img, caption="Uploaded Artifact", use_container_width=True)
        if st.button("🔎 Analyze Image"):
            with st.spinner("Analyzing visual artifact..."):
                try:
                    files = {
                        "file": (uploaded_img.name, uploaded_img.getvalue(), uploaded_img.type)
                    }
                    data = {"prompt": custom_vision_prompt}
                    resp = requests.post(
                        f"{API_URL}/v1/vision/analyze", files=files, data=data, timeout=60
                    )
                    if resp.status_code == 200:
                        out = resp.json()
                        st.markdown(
                            f"### Historical Insights:\n{out.get('analysis', out.get('answer', ''))}"
                        )
                    else:
                        st.error(f"Vision analysis failed: {resp.text}")
                except Exception as exc:
                    st.error(f"Vision request error: {exc}")


# -------------------------------------------------------------
# Mode 5: Multi-Domain Agent Hub (Medicine, Law, Custom)
# -------------------------------------------------------------
elif mode == "🤖 Multi-Domain Agent Hub (Medicine, Law, Custom)":
    st.subheader("🤖 Domain-Agnostic Multi-Agent Hub")
    st.caption(
        "Spawn and execute specialized agents for ANY domain (Medicine, History, Law, Science, Local RAG) with distinct providers."
    )

    tab1, tab2 = st.tabs(["💬 Query Domain Agents", "➕ Spawn Custom Domain Agent"])

    with tab1:
        st.markdown("#### Active Registered Agents")
        agents_data = []
        try:
            r = requests.get(f"{API_URL}/v1/agents", timeout=5)
            if r.status_code == 200:
                agents_data = r.json()
        except Exception as e:
            st.warning(f"Could not fetch agents: {e}")

        if agents_data:
            agent_options = {
                f"{a['name']} ({a['role']}) - [{a.get('domain', 'general').upper()}]": a["agent_id"]
                for a in agents_data
            }
            selected_label = st.selectbox(
                "Select Agent to Converse With:", list(agent_options.keys())
            )
            selected_agent_id = agent_options[selected_label]

            agent_info = next(a for a in agents_data if a["agent_id"] == selected_agent_id)
            st.info(
                f"**Domain**: `{agent_info.get('domain', 'general')}` | **Model**: `{agent_info.get('model_used', 'default')}`"
            )

            agent_query = st.text_area("Question or Task for this Agent:", height=100)
            if st.button("🚀 Run Agent Task") and agent_query:
                with st.spinner(f"Agent {agent_info['name']} is generating response..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/v1/agents/{selected_agent_id}/run",
                            json={"query": agent_query},
                            timeout=60,
                        )
                        if resp.status_code == 200:
                            res_json = resp.json()
                            st.markdown(f"### {res_json.get('agent_name', 'Agent')} Response:\n")
                            st.markdown(res_json.get("output", ""))
                            st.caption(
                                f"Confidence: {res_json.get('confidence', 0.95)} | Model: {res_json.get('model_used')}"
                            )
                        else:
                            st.error(f"Error {resp.status_code}: {resp.text}")
                    except Exception as ex:
                        st.error(f"Execution failed: {ex}")
        else:
            st.info("No agents discovered. Spawn one below!")

    with tab2:
        st.markdown("#### Spawn a Specialized Domain Model")
        st.caption(
            "Create a domain agent using any cloud provider (Gemini, OpenAI) or offline local engine (Ollama, Local GGUF)."
        )

        c1, c2 = st.columns(2)
        with c1:
            new_agent_id = st.text_input("Unique Agent ID:", value="domain-expert-ai")
            new_agent_name = st.text_input("Persona Name:", value="ExpertGuide")
            new_agent_role = st.text_input("Role Title:", value="Domain Specialist")
            new_agent_domain = st.text_input("Domain:", value="history")
        with c2:
            new_provider = st.selectbox(
                "LLM Provider Engine (Cloud or Local):",
                ["gemini", "openai", "ollama", "local"],
                index=0,
            )
            new_model = st.text_input(
                "Model Identifier / Path (optional override):",
                value="",
                placeholder="e.g. gpt-4o, llama3, or ./models/model.gguf",
            )
            new_temp = st.slider("Temperature:", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

        new_prompt = st.text_area(
            "System Prompt / Behavioral Guardrails:",
            value="You are an expert specialist in your domain. Analyze user queries with structured, accurate reasoning.",
            height=120,
        )

        if st.button("✨ Create and Register Agent"):
            payload = {
                "agent_id": new_agent_id,
                "name": new_agent_name,
                "role": new_agent_role,
                "domain": new_agent_domain,
                "system_prompt": new_prompt,
                "provider": new_provider,
                "model_name": new_model if new_model else None,
                "temperature": new_temp,
            }
            with st.spinner("Registering custom domain agent..."):
                try:
                    create_resp = requests.post(
                        f"{API_URL}/v1/agents/custom", json=payload, timeout=10
                    )
                    if create_resp.status_code == 200:
                        st.success(
                            f"Agent '{new_agent_name}' ({new_agent_id}) registered in domain '{new_agent_domain}' using provider '{new_provider}'!"
                        )
                        st.rerun()
                    else:
                        st.error(f"Failed to register agent: {create_resp.text}")
                except Exception as ex:
                    st.error(f"Creation failed: {ex}")
