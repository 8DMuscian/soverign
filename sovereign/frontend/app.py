"""Sovereign AI — Streamlit Frontend Dashboard.

Run with:
    streamlit run sovereign/frontend/app.py --server.port 8501

Multi-page dashboard for agent selection, file upload, code review,
security scanning, execution results, and file download.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import streamlit as st

# Add parent directory to path so we can import sovereign
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sovereign.agents import AgentRegistry
from sovereign.models import ModelRegistry
from sovereign.security.scanner import scan_code
from sovereign.vision import is_image_file
from sovereign import sanitize_ssl_env

# ────────────────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sovereign AI Workbench",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

TMP_ROOT = Path(__file__).parent.parent.parent / "sovereign_frontend_tmp"

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


@st.cache_data(ttl=30)
def query_server_cached(base_url: str) -> list[str]:
    """Cache /v1/models queries to avoid hammering vLLM on every rerun."""
    return ModelRegistry.query_server(base_url)

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
if "workspace_files" not in st.session_state:
    st.session_state.workspace_files = {}   # original filename -> tmp path
if "last_upload_name" not in st.session_state:
    st.session_state.last_upload_name = None
if "last_processed_file" not in st.session_state:
    st.session_state.last_processed_file = ""

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

    # Model info + reachability
    st.subheader("Model")
    default_model = model_registry.get_default()
    if default_model:
        st.write(f"**{default_model.name}**")
        st.caption(default_model.id)
        st.caption(f"Endpoint: {default_model.base_url}")

        served = query_server_cached(default_model.base_url)
        if served:
            if default_model.id not in served:
                best = model_registry.best_match(served, default_model.id)
                st.warning(f"Serving: {', '.join(served[:3])}")
                if best:
                    st.info(f"Will match to: '{best}'")
            else:
                st.success("Endpoint reachable, model ready")
        else:
            st.warning("⚠️ Endpoint unreachable — check Node 1")

    st.divider()

    # Settings
    st.subheader("Settings")
    sandbox_timeout = st.slider("Sandbox timeout (s)", 10, 300, 60)
    st.caption("Code runs automatically in an isolated sandbox. Review results below.")

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

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload file",
            type=[ext.lstrip(".") for ext in selected_agent.supported_file_types]
            if selected_agent.supported_file_types
            else None,
            help=f"Supported: {', '.join(selected_agent.supported_file_types)}",
            key="file_uploader",
        )

    with col2:
        if selected_agent.requires_vision:
            image_file = st.file_uploader(
                "Upload image",
                type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
                help="Image to analyze",
                key="image_uploader",
            )
        else:
            image_file = None

    if st.session_state.workspace_files:
        st.caption(
            f"Workspace: {', '.join(st.session_state.workspace_files.keys())} "
            "(edits persist across prompts)"
        )
        if st.button("🔄 Reset workspace (re-upload originals)"):
            for p in st.session_state.workspace_files.values():
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError:
                    pass
            st.session_state.workspace_files = {}
            st.session_state.last_upload_name = None
            st.session_state.generated_code = ""
            st.session_state.execution_result = None
            st.rerun()

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
                # ── Stage files in a persistent workspace ──
                TMP_ROOT.mkdir(parents=True, exist_ok=True)
                file_path = ""

                if uploaded_file and not selected_agent.is_registry_agent:
                    fname = uploaded_file.name
                    if st.session_state.last_upload_name != fname:
                        # New upload → save original bytes
                        tmp_path = TMP_ROOT / fname
                        tmp_path.write_bytes(uploaded_file.read())
                        st.session_state.workspace_files[fname] = str(tmp_path)
                        st.session_state.last_upload_name = fname
                    else:
                        # Same file already in workspace → keep prior edits
                        tmp_path = Path(
                            st.session_state.workspace_files.get(fname, TMP_ROOT / fname)
                        )
                    file_path = str(tmp_path)

                # Image upload (regardless of workspace persistence)
                image_path = ""
                if image_file:
                    img_path = TMP_ROOT / image_file.name
                    img_path.write_bytes(image_file.read())
                    image_path = str(img_path)

                # ── Build and run graph ──────────────────────
                from dotenv import load_dotenv
                load_dotenv()
                sanitize_ssl_env()

                from langchain_openai import ChatOpenAI
                from sovereign.graph import app
                from sovereign.nodes import ContextSchema

                model_config = None
                if selected_agent.model:
                    model_config = model_registry.get_model(selected_agent.model)
                if not model_config:
                    model_config = model_registry.get_default()

                if not model_config:
                    st.error("No model available. Check sovereign/models/ directory.")
                else:
                    # Auto-match model ID against what the server actually serves
                    served = query_server_cached(model_config.base_url)
                    if served:
                        best = model_registry.best_match(served, model_config.id)
                        if best and best != model_config.id:
                            st.warning(
                                f"Model '{model_config.id}' not found on server; "
                                f"using '{best}'."
                            )
                            model_config.id = best
                        elif not best:
                            st.error(
                                f"Model '{model_config.id}' is not served. "
                                f"Server has: {', '.join(served)}"
                            )

                    llm = ChatOpenAI(
                        model=model_config.id,
                        base_url=model_config.base_url,
                        api_key=model_config.api_key,
                        temperature=model_config.temperature,
                        max_tokens=model_config.max_tokens,
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
                        "auto": True,  # frontend executes in sandbox automatically
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
                        st.session_state.last_processed_file = file_path

                        st.session_state.history.append({
                            "agent": selected_agent.name,
                            "prompt": user_prompt,
                            "file": uploaded_file.name if uploaded_file else "(registry)",
                            "success": result.get("sandbox_result", {}).get("exit_code", -1) == 0,
                        })

                        st.rerun()
                    except Exception as exc:
                        st.error(f"Error: {exc}")

    # ── Execution results panel (rendered on every rerun) ──
    with st.container(border=True):
        st.subheader("📊 Execution Result")

        exec_result = st.session_state.execution_result
        if exec_result is None:
            st.caption("Nothing run yet — upload a file, enter a prompt, and hit Generate.")
        else:
            exit_code = exec_result.get("exit_code", -1)
            stdout = exec_result.get("stdout", "")
            stderr = exec_result.get("stderr", "")

            if exit_code == 0:
                st.success(f"✅ Execution succeeded — file modified in the sandbox.")
            else:
                st.error(f"❌ Execution failed (exit code {exit_code}).")

            if stdout:
                st.markdown("**Sandbox output:**")
                st.code(stdout)
            if stderr:
                st.markdown("**Errors:**")
                st.code(stderr, language="bash")
            if not stdout and not stderr:
                st.caption("No output produced.")

            # Download / save the modified file
            processed = st.session_state.last_processed_file
            if processed and Path(processed).exists() and Path(processed).is_file():
                size = Path(processed).stat().st_size
                st.caption(f"Modified file: `{processed}` ({size:,} bytes)")

                with open(processed, "rb") as fh:
                    st.download_button(
                        "📥 Download modified file",
                        data=fh.read(),
                        file_name=Path(processed).name,
                        mime="application/octet-stream",
                        use_container_width=True,
                    )

                save_dir = st.text_input(
                    "Optional: save a copy to a folder on Node 2 (server path)",
                    placeholder="e.g. C:/Users/you/Desktop",
                )
                if st.button("💾 Save copy to folder", use_container_width=True) and save_dir:
                    target = Path(save_dir)
                    try:
                        target.mkdir(parents=True, exist_ok=True)
                        dest = target / Path(processed).name
                        shutil.copy(processed, dest)
                        st.success(f"Saved to {dest}")
                    except OSError as exc:
                        st.error(f"Could not save: {exc}")

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
                st.session_state.execution_result = None
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
                st.write(f"**Max tokens:** {model.max_tokens}")
                st.write(f"**Temperature:** {model.temperature}")

                served = query_server_cached(model.base_url)
                if served:
                    st.write(f"**Serving on server:** {', '.join(served)}")
                    if model.id not in served:
                        best = model_registry.best_match(served, model.id)
                        st.warning(f"'id' mismatch — server match: {best}")
                else:
                    st.warning("Endpoint unreachable")
    else:
        st.warning("No models registered. Add YAML files to sovereign/models/.")

    st.divider()
    st.subheader("Add Model")
    with st.form("add_model"):
        col1, col2 = st.columns(2)
        with col1:
            new_name = st.text_input("Model Name")
            new_id = st.text_input("Model ID (HuggingFace or local)")
            new_base_url = st.text_input("Base URL", value="http://192.168.0.116:8000/v1")
        with col2:
            new_desc = st.text_input("Description")
            new_capabilities = st.multiselect(
                "Capabilities",
                ["code_generation", "vision", "image_understanding", "text_generation"],
                default=["code_generation"],
            )
            new_vision = st.checkbox("Requires Vision")
            new_default = st.checkbox("Set as Default")
            new_max_tokens = st.number_input("Max tokens", min_value=128, value=2048)

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
                "max_tokens": int(new_max_tokens),
                "temperature": 0.1,
                "max_model_len": 4096,
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