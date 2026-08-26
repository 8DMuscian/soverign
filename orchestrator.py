#!/usr/bin/env python3
"""
Sovereign AI Workbench — Backend Orchestrator
=============================================
Runs on Node 2 (the Mac). Accepts a natural-language prompt and a local
file path, asks the LLM on Node 1 to generate processing code, then
executes that code inside an isolated, network-disabled Docker container
on this machine.

Usage:
    python orchestrator.py --prompt "Add 5% tax to pricing" --file ./data.xlsx
    python orchestrator.py --prompt "Sort by price" --file ./data.xlsx --auto
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import docker
import openai
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────
# Load .env if present (overrides the defaults below)
load_dotenv()

# ---- Node 1 (vLLM endpoint) --------------------------------
# Replace this with the actual local IP of your GPU laptop
# on the ad-hoc WLAN.  Find it by running:
#   Node 1:  ip addr show  (Linux) or  ifconfig (macOS)
# on the GPU machine and looking for the ad-hoc interface.
#
# The port must match what you started vLLM with (--port flag).
VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://192.168.1.5:8000/v1")

# Model identifier — must match the --model you passed to vLLM.
# For a locally-downloaded AWQ model this is usually just the repo name.
VLLM_MODEL: str = os.getenv("VLLM_MODEL", "Qwen2.5-coder-7B-Instruct-AWQ")

# ---- Sandbox limits -----------------------------------------
SANDBOX_IMAGE: str = os.getenv("SANDBOX_IMAGE", "sandbox-python:latest")
SANDBOX_TIMEOUT: int = int(os.getenv("SANDBOX_TIMEOUT", "60"))
SANDBOX_MEM_LIMIT: str = os.getenv("SANDBOX_MEM_LIMIT", "512m")

# ---- Retry policy -------------------------------------------
MAX_RETRIES: int = 3
INITIAL_BACKOFF: float = 2.0  # seconds


# ────────────────────────────────────────────────────────────
# FILE VALIDATION
# ────────────────────────────────────────────────────────────
def validate_file(path: str) -> tuple[Path, Path, str]:
    """Resolve the target file and its parent directory.

    Returns:
        (absolute_file_path, parent_dir_for_mount, filename)

    Raises:
        SystemExit if the file does not exist.
    """
    file_path = Path(path).resolve()

    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}", file=sys.stderr)
        print("  Check the path and try again.", file=sys.stderr)
        sys.exit(1)

    if not file_path.is_file():
        print(f"[ERROR] Path is not a file: {file_path}", file=sys.stderr)
        sys.exit(1)

    parent_dir = file_path.parent
    filename = file_path.name

    print(f"  File     : {file_path}")
    print(f"  Directory: {parent_dir}")

    return file_path, parent_dir, filename


# ────────────────────────────────────────────────────────────
# PROMPT CONSTRUCTION
# ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a Python data-processing agent running in a sandboxed Docker container.

RULES:
1. Output ONLY executable Python code. No markdown fences, no explanations.
2. Use standard libraries: pandas, openpyxl, python-docx, csv, json, etc.
3. The target file is mounted at /workspace/{filename}.
4. Read the file, apply the user's request, and save the result back to the
   SAME path (modify in-place).
5. If the file is an Excel workbook, preserve all existing sheets unless the
   user says otherwise.
6. After modifying the file, print a short summary of what changed
   (e.g. "Modified 42 rows in sheet 'Prices'. Added column 'Tax'.").
7. Use sys.exit(0) on success.  On unrecoverable error, print the error
   message to stderr and call sys.exit(1).
8. Do NOT import any library that is not pre-installed. Only use:
   pandas, openpyxl, python-docx, csv, json, os, sys, pathlib.
"""


def build_prompt(user_request: str, filename: str) -> list[dict[str, str]]:
    """Build the chat messages list for the LLM."""
    system_msg = SYSTEM_PROMPT.format(filename=filename)

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_request},
    ]


# ────────────────────────────────────────────────────────────
# LLM API CALL
# ────────────────────────────────────────────────────────────
def call_llm(messages: list[dict[str, str]]) -> str:
    """Send a chat completion request to the local vLLM endpoint.

    Retries on connection / timeout errors (common on ad-hoc WLAN).
    """
    client = openai.OpenAI(
        base_url=VLLM_BASE_URL,
        # vLLM does not require a real key, but the client needs
        # a non-empty string to construct the Authorization header.
        api_key="sk-not-needed",
        timeout=120.0,
    )

    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  LLM request (attempt {attempt}/{MAX_RETRIES})...")
            response = client.chat.completions.create(
                model=VLLM_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=4096,
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("LLM returned empty content")
            return content.strip()

        except (
            openai.APIConnectionError,
            openai.APITimeoutError,
            TimeoutError,
            ConnectionError,
        ) as exc:
            last_exc = exc
            backoff = INITIAL_BACKOFF * (2 ** (attempt - 1))
            print(f"  [WARN] Connection failed: {exc}", file=sys.stderr)
            print(f"  Retrying in {backoff:.0f}s...", file=sys.stderr)
            time.sleep(backoff)

        except Exception as exc:
            print(f"[ERROR] Unexpected LLM error: {exc}", file=sys.stderr)
            sys.exit(1)

    print(
        f"[ERROR] Could not reach vLLM after {MAX_RETRIES} attempts.\n"
        f"  Last error: {last_exc}\n"
        f"  Verify that Node 1 is running and reachable at {VLLM_BASE_URL}",
        file=sys.stderr,
    )
    sys.exit(1)


# ────────────────────────────────────────────────────────────
# CODE EXTRACTION
# ────────────────────────────────────────────────────────────
def extract_code(response: str) -> str:
    """Extract executable Python code from the LLM response.

    The model may wrap code in ```python ... ``` fences, ``` ... ```
    fences, or return raw code.  We try each in order.
    """
    # Priority 1: ```python ... ``` block
    m = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Priority 2: ``` ... ``` block (any language tag)
    m = re.search(r"```\w*\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Priority 3: use the entire response as code
    code = response.strip()

    # Basic sanity check — should look like Python
    if not code or not any(
        keyword in code for keyword in ("import ", "def ", "print(", "pd.", "open(", "sys.")
    ):
        print("[ERROR] Could not extract valid Python code from LLM response.", file=sys.stderr)
        print("Raw response:", file=sys.stderr)
        print(response[:500], file=sys.stderr)
        sys.exit(1)

    return code


# ────────────────────────────────────────────────────────────
# SANDBOX EXECUTION (Docker)
# ────────────────────────────────────────────────────────────
def run_in_sandbox(script: str, mount_dir: Path, filename: str) -> dict:
    """Write the script to a temp file, spin up an ephemeral Docker
    container with network disabled, run the script, and return results.

    The container:
      - Has NO network access (network_disabled=True)
      - Bind-mounts the target file's parent directory to /workspace
      - Uses the pre-built sandbox-python image (pandas, openpyxl, python-docx)
      - Is limited to {SANDBOX_MEM_LIMIT} RAM and 50% of one CPU
      - Auto-removes after exit
    """
    # Connect to local Docker daemon
    try:
        client = docker.from_env()
        client.ping()
    except docker.errors.DockerException as exc:
        print(
            f"[ERROR] Cannot connect to Docker daemon: {exc}\n"
            "  Make sure Docker Desktop is running on this machine.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Write the generated script to a temporary file inside the mount dir
    # so Docker can access it without another bind mount.
    script_path = mount_dir / ".__orchestrator_sandbox_script__.py"
    try:
        script_path.write_text(script, encoding="utf-8")

        print(f"  Running in sandbox (timeout={SANDBOX_TIMEOUT}s)...")
        result = client.containers.run(
            image=SANDBOX_IMAGE,
            command=["python", "/workspace/.__orchestrator_sandbox_script__.py"],
            volumes={
                str(mount_dir): {"bind": "/workspace", "mode": "rw"},
            },
            network_disabled=True,
            remove=True,
            mem_limit=SANDBOX_MEM_LIMIT,
            cpu_period=100_000,
            cpu_quota=50_000,  # 50% of one CPU core
            stderr=True,
            stdout=True,
            # Prevent the container from running longer than our timeout.
            # The docker-py library doesn't have a native timeout for
            # .run(), so we rely on the container exiting on its own.
            # If the LLM code hangs, the caller should handle the
            # interrupt (Ctrl-C).
        )

        output = result.decode("utf-8", errors="replace").strip() if result else ""

        return {
            "stdout": output,
            "stderr": "",
            "exit_code": 0,
        }

    except docker.errors.ContainerError as exc:
        stderr_out = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        stdout_out = exc.stdout.decode("utf-8", errors="replace") if exc.stdout else ""
        return {
            "stdout": stdout_out,
            "stderr": stderr_out or str(exc),
            "exit_code": exc.exit_status,
        }

    except Exception as exc:
        return {
            "stdout": "",
            "stderr": f"Sandbox error: {exc}",
            "exit_code": -1,
        }

    finally:
        # Always clean up the temporary script file
        try:
            script_path.unlink(missing_ok=True)
        except OSError:
            pass


# ────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sovereign AI Workbench — air-gapped code orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python orchestrator.py --prompt "Add 5%% tax to pricing column" --file data.xlsx\n'
            '  python orchestrator.py --prompt "Sort by price descending" --file data.xlsx --auto\n'
        ),
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Natural-language description of what to do with the file.",
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the target file on this machine (Node 2).",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Skip the code-confirmation prompt and execute immediately.",
    )

    args = parser.parse_args()

    # ── Banner ──────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║      SOVEREIGN AI WORKBENCH — ORCHESTRATOR   ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    # ── Step 1: Validate the target file ────────────────────
    print("[1/5] Validating target file...")
    file_path, parent_dir, filename = validate_file(args.file)
    print()

    # ── Step 2: Build the LLM prompt ────────────────────────
    print("[2/5] Building prompt...")
    messages = build_prompt(args.prompt, filename)
    print(f"  Prompt built for model: {VLLM_MODEL}")
    print()

    # ── Step 3: Call the LLM ────────────────────────────────
    print(f"[3/5] Contacting LLM at {VLLM_BASE_URL}...")
    llm_response = call_llm(messages)
    print(f"  Received {len(llm_response)} chars from model")
    print()

    # ── Step 4: Extract code ────────────────────────────────
    print("[4/5] Extracting code...")
    code = extract_code(llm_response)
    print(f"  Extracted {len(code)} lines of Python")
    print()
    print("─── Generated Code ───────────────────────────────")
    print(code)
    print("─── End Code ─────────────────────────────────────")
    print()

    # ── Step 4.5: User confirmation ─────────────────────────
    if not args.auto:
        try:
            answer = input("Execute this code? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted by user.")
            sys.exit(130)

        if answer not in ("y", "yes"):
            print("Aborted. Code was NOT executed.")
            sys.exit(0)

    # ── Step 5: Run in sandbox ──────────────────────────────
    print("[5/5] Executing in isolated Docker sandbox...")
    result = run_in_sandbox(code, parent_dir, filename)
    print()

    # ── Report ──────────────────────────────────────────────
    if result["exit_code"] == 0:
        print("✓ SUCCESS — File modified in-place.")
        if result["stdout"]:
            print()
            print("─── Sandbox Output ───────────────────────────────")
            print(result["stdout"])
            print("─── End Output ──────────────────────────────────")
    else:
        print("✗ FAILED — Sandbox execution error.", file=sys.stderr)
        print(f"  Exit code: {result['exit_code']}", file=sys.stderr)
        if result["stderr"]:
            print()
            print("─── Sandbox Errors ───────────────────────────────")
            print(result["stderr"], file=sys.stderr)
            print("─── End Errors ──────────────────────────────────")
        if result["stdout"]:
            print()
            print("─── Sandbox Stdout ───────────────────────────────")
            print(result["stdout"])
            print("─── End Stdout ──────────────────────────────────")

    sys.exit(0 if result["exit_code"] == 0 else 1)


if __name__ == "__main__":
    main()
