"""Graph state definition for the Sovereign AI orchestrator."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import add_messages
from typing_extensions import Annotated


class SandboxResult(TypedDict, total=False):
    stdout: str
    stderr: str
    exit_code: int


class OrchestratorState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────
    file_path: str          # raw path from CLI
    file_dir: str           # resolved parent directory (for Docker mount)
    filename: str           # basename of the target file
    user_prompt: str        # natural-language instruction
    agent_name: str         # selected agent name
    image_path: str         # path to image file (for vision agents)

    # ── LLM interaction ───────────────────────────────────
    messages: Annotated[list, add_messages]  # chat history (auto-merges)
    code: str               # extracted executable Python
    llm_response: str       # raw LLM output

    # ── Security ──────────────────────────────────────────
    security_result: dict   # ScanResult as dict (passed, issues, summary)
    security_attempts: int  # how many times security scan has looped back

    # ── Retry bookkeeping ─────────────────────────────────
    extract_attempts: int   # how many times extract_code has run
    sandbox_attempts: int   # how many times sandbox has been attempted
    max_extract_retries: int
    max_sandbox_retries: int
    error_feedback: str     # latest error context for the LLM

    # ── Confirmation ──────────────────────────────────────
    auto: bool              # True = skip interactive confirmation
    approved: bool          # True = user approved code execution

    # ── Execution ─────────────────────────────────────────
    sandbox_result: SandboxResult
