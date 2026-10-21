"""HistoryVerse Streamlit UI.

Provides an interactive user interface for:
- HistoryVerse Chat (Persona Mora)
- HistoryVerse AR Mode (Statue & Artifact Guided Tour)
- Chat with Historical Documents
- Historical Vision & Artifact Analysis
"""

import os
import uuid

import requests
import streamlit as st

API_URL = os.getenv("HISTORYVERSE_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="HistoryVerse AI",
    page_icon="🏛️",
    layout="wide",
)

st.title("🏛️ HistoryVerse AI Portal")
st.markdown("Model-agnostic Historical AI Assistant & Augmented Reality Guide")

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
            "🏛️ HistoryVerse Chat",
            "🏺 HistoryVerse AR Mode",
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
# Mode 1: HistoryVerse Chat (Mora Persona)
# -------------------------------------------------------------
if mode == "🏛️ HistoryVerse Chat":
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
# Mode 2: HistoryVerse AR Mode (Statues & Artifacts)
# -------------------------------------------------------------
elif mode == "🏺 HistoryVerse AR Mode":
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
                    f"{API_URL}/v1/historyverse-ar-mode/start",
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
                    f"{API_URL}/v1/historyverse-ar-mode/chat",
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
