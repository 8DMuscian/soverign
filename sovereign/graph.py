"""LangGraph StateGraph definition for the Sovereign AI orchestrator.

Graph topology:

    START → validate_file → build_prompt → call_llm → extract_code
                                              ↑              │
                                              │    ┌─────────┼──────────┐
                                              │    │         │          │
                                              │ [retry]   [valid]   [failed]
                                              │    │         │          │
                                              │    └─────────┘          ▼
                                              │                    END (error)
                                              │
                                              │              security_scan
                                              │                    │
                                              │           ┌────────┼────────┐
                                              │           │        │        │
                                              │       [clean]  [critical]  [retry]
                                              │           │        │        │
                                              │           ▼        ▼        ▼
                                              │        confirm  inject_  call_llm
                                              │           │    security
                                              │           │     error
                                              │     ┌─────┼─────┐
                                              │ [approved] [rejected]
                                              │     │         │
                                              ▼     ▼         ▼
                                           END  run_sandbox  END
                                              or write_yaml
                                                  │
                                        [success] │  [fail + retries]
                                            │     │       │
                                            ▼     ▼       │
                                         END  (retry to call_llm)
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    build_prompt,
    call_llm,
    confirm,
    extract_code,
    run_sandbox,
    security_scan,
    validate_file,
    write_model_config,
)
from .state import OrchestratorState

# ── Retry limits (read from env at graph-build time) ───────
import os

MAX_EXTRACT_RETRIES = int(os.getenv("MAX_EXTRACT_RETRIES", "3"))
MAX_SANDBOX_RETRIES = int(os.getenv("MAX_SANDBOX_RETRIES", "2"))
MAX_SECURITY_RETRIES = int(os.getenv("MAX_SECURITY_RETRIES", "2"))


# ────────────────────────────────────────────────────────────
# CONDITIONAL EDGE FUNCTIONS
# ────────────────────────────────────────────────────────────

def after_extract(state: OrchestratorState) -> str:
    """Route after code extraction.

    - If code was extracted → proceed to security scan.
    - If extraction failed but retries remain → loop back to call_llm.
    - If retries exhausted → end with error.
    """
    if state.get("code"):
        return "security_scan"

    attempts = state.get("extract_attempts", 0)
    if attempts < MAX_EXTRACT_RETRIES:
        print(
            f"  Retrying LLM call ({attempts}/{MAX_EXTRACT_RETRIES})..."
        )
        return "retry_llm"

    return "extraction_failed"


def after_security(state: OrchestratorState, runtime=None) -> str:
    """Route after security scan.

    - If passed → proceed to confirm (code agents) or write_yaml (registry agents).
    - If critical issues and retries remain → loop back to call_llm.
    - If critical issues and no retries → end with error.
    """
    result = state.get("security_result", {})

    if result.get("passed", False):
        # Check if this is a registry agent → skip sandbox, write YAML directly
        agent_config = None
        if runtime and hasattr(runtime, "context"):
            agent_config = runtime.context.agent_config
        if agent_config and getattr(agent_config, "is_registry_agent", False):
            return "write_yaml"
        return "confirm"

    # Security issues found — check retries
    attempts = state.get("security_attempts", 0)
    if attempts < MAX_SECURITY_RETRIES:
        print(
            f"  Security issues found — retrying LLM "
            f"({attempts}/{MAX_SECURITY_RETRIES})..."
        )
        return "retry_security"

    return "security_failed"


def after_confirm(state: OrchestratorState) -> str:
    """Route after user confirmation.

    - Approved → run in sandbox.
    - Rejected or no code → end.
    """
    if state.get("approved") and state.get("code"):
        return "run_sandbox"
    return "aborted"


def after_sandbox(state: OrchestratorState) -> str:
    """Route after sandbox execution.

    - Success → end.
    - Failure with retries remaining → loop back to call_llm with error.
    - Failure with no retries → end with error.
    """
    result = state.get("sandbox_result", {})
    if result.get("exit_code", -1) == 0:
        return "success"

    attempts = state.get("sandbox_attempts", 0)
    if attempts < MAX_SANDBOX_RETRIES:
        print(
            f"  Sandbox failed — retrying LLM "
            f"({attempts}/{MAX_SANDBOX_RETRIES})..."
        )
        return "retry_llm_sandbox"

    return "sandbox_failed"


def inject_sandbox_error(state: OrchestratorState) -> dict:
    """Prepend an error-feedback message to the chat history so the
    LLM knows what went wrong on the previous sandbox run."""
    result = state.get("sandbox_result", {})
    stderr = result.get("stderr", "Unknown error")
    stdout = result.get("stdout", "")
    exit_code = result.get("exit_code", -1)

    feedback = (
        f"Your previous code execution failed.\n"
        f"Exit code: {exit_code}\n"
        f"Stderr:\n{stderr}\n"
    )
    if stdout:
        feedback += f"Stdout:\n{stdout}\n"

    feedback += (
        "\nPlease fix the code and output ONLY valid Python. "
        "No markdown, no explanations."
    )

    from langchain_core.messages import HumanMessage

    return {
        "error_feedback": feedback,
        "messages": [HumanMessage(content=feedback)],
    }


def inject_extract_error(state: OrchestratorState) -> dict:
    """Prepend an extraction-failure feedback message to chat history."""
    feedback = state.get("error_feedback", "Could not extract code.")

    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content=feedback)],
    }


def inject_security_error(state: OrchestratorState) -> dict:
    """Prepend a security-scan failure feedback message to chat history."""
    result = state.get("security_result", {})
    issues = result.get("issues", [])

    feedback_parts = [
        "Your generated code failed the security scan.",
        "Security issues found:",
    ]
    for issue in issues:
        severity = issue.get("severity", "unknown")
        message = issue.get("message", "")
        suggestion = issue.get("suggestion", "")
        feedback_parts.append(f"  [{severity.upper()}] {message}")
        if suggestion:
            feedback_parts.append(f"    Fix: {suggestion}")

    feedback_parts.append(
        "\nPlease regenerate the code avoiding these security issues. "
        "Output ONLY valid Python. No markdown, no explanations."
    )

    feedback = "\n".join(feedback_parts)
    attempt = state.get("security_attempts", 0) + 1

    from langchain_core.messages import HumanMessage

    return {
        "error_feedback": feedback,
        "security_attempts": attempt,
        "messages": [HumanMessage(content=feedback)],
    }


# ────────────────────────────────────────────────────────────
# GRAPH BUILDER
# ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construct and compile the orchestrator StateGraph."""
    graph = StateGraph(OrchestratorState)

    # ── Add nodes ──────────────────────────────────────────
    graph.add_node("validate_file", validate_file)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("call_llm", call_llm)
    graph.add_node("extract_code", extract_code)
    graph.add_node("security_scan", security_scan)
    graph.add_node("confirm", confirm)
    graph.add_node("run_sandbox", run_sandbox)
    graph.add_node("write_model_config", write_model_config)

    # Retry helper nodes (inject feedback then go back to call_llm)
    graph.add_node("inject_extract_error", inject_extract_error)
    graph.add_node("inject_sandbox_error", inject_sandbox_error)
    graph.add_node("inject_security_error", inject_security_error)

    # ── Linear path ────────────────────────────────────────
    graph.add_edge(START, "validate_file")
    graph.add_edge("validate_file", "build_prompt")
    graph.add_edge("build_prompt", "call_llm")
    graph.add_edge("call_llm", "extract_code")

    # ── After extract → security scan ──────────────────────
    graph.add_conditional_edges(
        "extract_code",
        after_extract,
        {
            "security_scan": "security_scan",
            "retry_llm": "inject_extract_error",
            "extraction_failed": END,
        },
    )

    # After injecting extract error → go back to call_llm
    graph.add_edge("inject_extract_error", "call_llm")

    # ── After security → confirm, write_yaml, retry, or fail ──
    graph.add_conditional_edges(
        "security_scan",
        after_security,
        {
            "confirm": "confirm",
            "write_yaml": "write_model_config",
            "retry_security": "inject_security_error",
            "security_failed": END,
        },
    )

    # After injecting security error → go back to call_llm
    graph.add_edge("inject_security_error", "call_llm")

    # ── After write_model_config → end ─────────────────────
    graph.add_edge("write_model_config", END)

    # ── Confirm → sandbox or abort ─────────────────────────
    graph.add_conditional_edges(
        "confirm",
        after_confirm,
        {
            "run_sandbox": "run_sandbox",
            "aborted": END,
        },
    )

    # ── Sandbox → success, retry, or fail ──────────────────
    graph.add_conditional_edges(
        "run_sandbox",
        after_sandbox,
        {
            "success": END,
            "retry_llm_sandbox": "inject_sandbox_error",
            "sandbox_failed": END,
        },
    )

    # After injecting sandbox error → go back to call_llm
    graph.add_edge("inject_sandbox_error", "call_llm")

    return graph.compile()


# ────────────────────────────────────────────────────────────
# COMPILED GRAPH (importable singleton)
# ────────────────────────────────────────────────────────────
app = build_graph()
