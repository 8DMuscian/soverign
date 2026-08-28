"""Sovereign AI — Streamlit Frontend Dashboard.

Run with:
    streamlit run sovereign/frontend/app.py --server.port 8501

Multi-page dashboard for agent selection, file upload, code review,
security scanning, and execution logs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Add parent directory to path so we can import sovereign
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sovereign.agents import AgentRegistry
from sovereign.models import ModelRegistry
from sovereign.security.scanner import scan_code
from sovereign.vision import is_image_file

# ────────────────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sovereign AI Workbench",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────
# INIT REGISTRIES
# ────────────────────────────────────────────────────────────
@st.cache_resource
def load_registries():
    base = Path(__file__).parent.parent
    agents = AgentRegistry(base / "agents")
    models = ModelRegistry(base / "models")
    return agents, models

agent_registry, model_registry = load_registries()

# ────────────────────────────────────────────────────────────
# SESSION STATE
# ────────────────────────────────────────────────────────────
if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""
if "security_result" not in st.session_state:
    st.session_state.security_result = None
if "execution_result" not in st.session_state:
    st.session_state.execution_result = None
if "history" not in st.session_state:
    st.session_state.history = []

# ────────────────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Sovereign AI")
    st.caption("LangGraph Multi-Agent Orchestrator")

    st.divider()

    # Agent selection
    st.subheader("Agent")
    agents = agent_registry.list_agents()
    agent_names = [f"{a.icon} {a.name}" for a in agents]
    selected_idx = st.selectbox(
        "Select Agent",
        range(len(agent_names)),
        format_func=lambda i: agent_names[i],
    )
    selected_agent = agents[selected_idx]

    # Show agent info
    st.caption(selected_agent.description)
    if selected_agent.requires_vision:
        st.info("📷 This agent uses a vision model")
    if selected_agent.model:
        st.caption(f"Model: {selected_agent.model}")

    st.divider()

    # Model info
    st.subheader("Model")
    default_model = model_registry.get_default()
    if default_model:
        st.write(f"**{default_model.name}**")
        st.caption(default_model.id)
        st.caption(f"Endpoint: {default_model.base_url}")

    st.divider()

    # Settings
    st.subheader("Settings")
    auto_mode = st.checkbox("Auto-execute (skip confirmation)", value=False)
    sandbox_timeout = st.slider("Sandbox timeout (s)", 10, 300, 60)

# ────────────────────────────────────────────────────────────
# MAIN CONTENT — TABS
# ────────────────────────────────────────────────────────────
tab_work, tab_code, tab_security, tab_models, tab_history = st.tabs(
    ["📝 Work", "💻 Code Review", "🔒 Security", "📦 Models", "📜 History"]
)

# ────────────────────────────────────────────────────────────
# TAB: WORK
# ────────────────────────────────────────────────────────────
with tab_work:
    st.header(f"{selected_agent.icon} {selected_agent.name}")

    # File upload
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload file",
            type=[ext.lstrip(".") for ext in selected_agent.supported_file_types]
            if selected_agent.supported_file_types
            else None,
            help=f"Supported: {', '.join(selected_agent.supported_file_types)}",
        )

    with col2:
        # Image upload for vision agents
        if selected_agent.requires_vision:
            image_file = st.file_uploader(
                "Upload image",
                type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
                help="Image to analyze",
            )
        else:
            image_file = None

    # Prompt
    user_prompt = st.text_area(
        "Describe what you want to do",
        placeholder="e.g., Add 5% tax to the pricing column...",
        height=100,
    )

    # Generate button
    if st.button("🚀 Generate", type="primary", use_container_width=True):
        if not uploaded_file and not selected_agent.is_registry_agent:
            st.error("Please upload a file first.")
        elif not user_prompt:
            st.error("Please enter a prompt.")
        else:
            with st.spinner("Generating code with LLM..."):
                # Save uploaded file temporarily
                if uploaded_file:
                    tmp_dir = Path("sovereign_frontend_tmp")
                    tmp_dir.mkdir(exist_ok=True)
                    tmp_path = tmp_dir / uploaded_file.name
                    tmp_path.write_bytes(uploaded_file.read())
                    file_path = str(tmp_path)
                else:
                    file_path = ""

                # Save image if present
                image_path = ""
                if image_file:
                    img_dir = Path("sovereign_frontend_tmp")
                    img_dir.mkdir(exist_ok=True)
                    img_path = img_dir / image_file.name
                    img_path.write_bytes(image_file.read())
                    image_path = str(img_path)

                # Build and run graph
                from dotenv import load_dotenv
                load_dotenv()

                from langchain_openai import ChatOpenAI
                from sovereign.graph import app
                from sovereign.nodes import ContextSchema

                # Resolve model
                model_config = None
                if selected_agent.model:
                    model_config = model_registry.get_model(selected_agent.model)
                if not model_config:
                    model_config = model_registry.get_default()

                if not model_config:
                    st.error("No model available. Check sovereign/models/ directory.")
                else:
                    llm = ChatOpenAI(
                        model=model_config.id,
                        base_url=model_config.base_url,
                        api_key=model_config.api_key,
                        temperature=0.1,
                        max_tokens=4096,
                        timeout=120,
                    )

                    ctx = ContextSchema(
                        llm=llm,
                        model_config=model_config,
                        agent_config=selected_agent,
                        sandbox_image=os.getenv("SANDBOX_IMAGE", "sandbox-python:latest"),
                        sandbox_timeout=sandbox_timeout,
                        sandbox_mem_limit=os.getenv("SANDBOX_MEM_LIMIT", "512m"),
                    )

                    initial_state = {
                        "user_prompt": user_prompt,
                        "auto": True,  # Frontend always auto (we show code for review)
                        "extract_attempts": 0,
                        "sandbox_attempts": 0,
                        "security_attempts": 0,
                        "agent_name": selected_agent.name,
                    }

                    if file_path:
                        initial_state["file_path"] = file_path
                    if image_path:
                        initial_state["image_path"] = image_path

                    try:
                        result = app.invoke(initial_state, context=ctx)
                        st.session_state.generated_code = result.get("code", "")
                        st.session_state.security_result = result.get("security_result")
                        st.session_state.execution_result = result.get("sandbox_result")

                        # Add to history
                        st.session_state.history.append({
                            "agent": selected_agent.name,
                            "prompt": user_prompt,
                            "file": uploaded_file.name if uploaded_file else "(registry)",
                            "success": result.get("sandbox_result", {}).get("exit_code", -1) == 0,
                        })

                        st.success("Generation complete! Check the Code Review tab.")
                    except Exception as exc:
                        st.error(f"Error: {exc}")

# ────────────────────────────────────────────────────────────
# TAB: CODE REVIEW
# ────────────────────────────────────────────────────────────
with tab_code:
    st.header("Generated Code")

    code = st.session_state.generated_code
    if code:
        st.code(code, language="python", line_numbers=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Copy Code"):
                st.toast("Code copied to clipboard!")
        with col2:
            if st.button("🔄 Re-generate"):
                st.session_state.generated_code = ""
                st.rerun()
    else:
        st.info("No code generated yet. Use the Work tab to generate code.")

# ────────────────────────────────────────────────────────────
# TAB: SECURITY
# ────────────────────────────────────────────────────────────
with tab_security:
    st.header("Security Scan Results")

    result = st.session_state.security_result
    if result:
        passed = result.get("passed", False)
        issues = result.get("issues", [])
        summary = result.get("summary", "")

        if passed:
            st.success(f"✅ {summary}")
        else:
            st.error(f"❌ {summary}")

        if issues:
            for issue in issues:
                severity = issue.get("severity", "info")
                category = issue.get("category", "")
                line = issue.get("line")
                message = issue.get("message", "")
                suggestion = issue.get("suggestion", "")

                icon = {"critical": "🔴", "warning": "🟡", "info": "🟢"}.get(severity, "⚪")
                line_str = f" (line {line})" if line else ""

                with st.expander(f"{icon} {severity.upper()}: {message}{line_str}"):
                    st.write(f"**Category:** {category}")
                    st.write(f"**Suggestion:** {suggestion}")
    else:
        st.info("No security scan results yet. Generate code first.")

# ────────────────────────────────────────────────────────────
# TAB: MODELS
# ────────────────────────────────────────────────────────────
with tab_models:
    st.header("Registered Models")

    models = model_registry.list_models()
    if models:
        for model in models:
            with st.expander(
                f"{'⭐ ' if model.default else ''}{model.name} — {model.id}"
            ):
                st.write(f"**Description:** {model.description}")
                st.write(f"**Endpoint:** `{model.base_url}`")
                st.write(f"**Capabilities:** {', '.join(model.capabilities)}")
                st.write(f"**Vision:** {'Yes' if model.requires_vision else 'No'}")
                st.write(f"**Default:** {'Yes' if model.default else 'No'}")
    else:
        st.warning("No models registered. Add YAML files to sovereign/models/.")

    st.divider()
    st.subheader("Add Model")
    with st.form("add_model"):
        col1, col2 = st.columns(2)
        with col1:
            new_name = st.text_input("Model Name")
            new_id = st.text_input("Model ID (HuggingFace or local)")
            new_base_url = st.text_input("Base URL", value="http://192.168.1.5:8000/v1")
        with col2:
            new_desc = st.text_input("Description")
            new_capabilities = st.multiselect(
                "Capabilities",
                ["code_generation", "vision", "image_understanding", "text_generation"],
                default=["code_generation"],
            )
            new_vision = st.checkbox("Requires Vision")
            new_default = st.checkbox("Set as Default")

        submitted = st.form_submit_button("Add Model")
        if submitted and new_name and new_id:
            import yaml
            data = {
                "name": new_name,
                "id": new_id,
                "description": new_desc,
                "base_url": new_base_url,
                "api_key": "not-needed",
                "capabilities": new_capabilities,
                "requires_vision": new_vision,
                "default": new_default,
            }
            slug = "".join(c if c.isalnum() else "-" for c in new_name.lower()).strip("-")
            filepath = Path(__file__).parent.parent / "models" / f"{slug}.yaml"
            with open(filepath, "w") as fh:
                yaml.dump(data, fh, default_flow_style=False)
            st.success(f"Model added: {filepath.name}")
            st.rerun()

# ────────────────────────────────────────────────────────────
# TAB: HISTORY
# ────────────────────────────────────────────────────────────
with tab_history:
    st.header("Session History")

    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history), 1):
            status = "✅" if entry["success"] else "❌"
            st.write(f"{status} **{entry['agent']}** — {entry['prompt'][:80]}...")
            st.caption(f"File: {entry['file']}")
    else:
        st.info("No history yet.")
