"""CLI entry point for the Sovereign AI orchestrator.

Usage:
    python -m sovereign.cli --prompt "Add 5% tax" --file ./data.xlsx
    python -m sovereign.cli --prompt "Sort by price" --file ./data.xlsx --auto
    python -m sovereign.cli --list-agents
    python -m sovereign.cli --list-models
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .graph import app
from .nodes import ContextSchema
from . import sanitize_ssl_env


def main() -> None:
    load_dotenv()
    sanitize_ssl_env()

    # ── Configuration from env ─────────────────────────────
    sandbox_image: str = os.getenv(
        "SANDBOX_IMAGE", "sandbox-python:latest"
    )
    sandbox_timeout: int = int(os.getenv("SANDBOX_TIMEOUT", "60"))
    sandbox_mem_limit: str = os.getenv("SANDBOX_MEM_LIMIT", "1g")

    # ── CLI arguments ──────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Sovereign AI Workbench — LangGraph-powered orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python -m sovereign.cli --prompt "Add 5%% tax to pricing" '
            '--file data.xlsx\n'
            '  python -m sovereign.cli --prompt "Sort by price" '
            '--file data.xlsx --auto\n'
            '  python -m sovereign.cli --list-agents\n'
            '  python -m sovereign.cli --list-models\n'
        ),
    )
    parser.add_argument(
        "--prompt",
        help="Natural-language description of what to do with the file.",
    )
    parser.add_argument(
        "--file",
        help="Path to the target file on this machine.",
    )
    parser.add_argument(
        "--agent",
        help="Agent name to use (e.g. 'Data Processor', 'Image Processor').",
    )
    parser.add_argument(
        "--image",
        help="Path to an image file (for vision agents).",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Skip the code-confirmation prompt and execute immediately.",
    )
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="List all registered agents and exit.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all registered models and exit.",
    )

    args = parser.parse_args()

    # ── Import registries ──────────────────────────────────
    from .agents import AgentRegistry
    from .models import ModelRegistry

    agent_registry = AgentRegistry()
    model_registry = ModelRegistry()

    # ── Handle list commands ───────────────────────────────
    if args.list_agents:
        print()
        print("=" * 50)
        print("  REGISTERED AGENTS")
        print("=" * 50)
        print()
        print(agent_registry.summary())
        print()
        sys.exit(0)

    if args.list_models:
        print()
        print("=" * 50)
        print("  REGISTERED MODELS")
        print("=" * 50)
        print()
        print(model_registry.summary())
        print()
        sys.exit(0)

    # ── Validate required args ─────────────────────────────
    if not args.prompt:
        parser.error("--prompt is required (or use --list-agents / --list-models)")

    # ── Resolve agent ──────────────────────────────────────
    agent_config = None
    if args.agent:
        agent_config = agent_registry.get_agent(args.agent)
        if not agent_config:
            print(f"[ERROR] Agent not found: {args.agent}", file=sys.stderr)
            print(f"Available agents:", file=sys.stderr)
            for a in agent_registry.list_agents():
                print(f"  - {a.name} ({a.category})", file=sys.stderr)
            sys.exit(1)
    else:
        # Default to first enabled agent (usually Data Processor)
        agents = agent_registry.list_agents()
        if agents:
            agent_config = agents[0]
            print(f"  Using default agent: {agent_config.name}")

    # ── Resolve model ──────────────────────────────────────
    model_config = None
    if agent_config and agent_config.model:
        model_config = model_registry.get_model(agent_config.model)
        if not model_config:
            print(
                f"[WARN] Model '{agent_config.model}' not found, using default",
                file=sys.stderr,
            )

    if not model_config:
        model_config = model_registry.get_default()

    if not model_config:
        print("[ERROR] No models registered. Add a model YAML to sovereign/models/", file=sys.stderr)
        sys.exit(1)

    # ── Verify endpoint & validate model ID ─────────────────
    served = model_registry.query_server(model_config.base_url)
    if not served:
        print(
            f"[WARN] Could not reach {model_config.base_url}\n"
            f"       Check Node 1 is up and the base_url in the model YAML is correct.",
            file=sys.stderr,
        )
    else:
        best = model_registry.best_match(served, model_config.id)
        if best and best != model_config.id:
            print(
                f"  Model ID '{model_config.id}' not found on server.\n"
                f"  Server serves: {', '.join(served)}\n"
                f"  Using best match: {best}",
                file=sys.stderr,
            )
            model_config.id = best
        elif not best:
            print(
                f"[WARN] Model '{model_config.id}' not found on server.\n"
                f"       Server serves: {', '.join(served)}\n"
                f"       Update the 'id' field in the model YAML, or pass "
                f"--served-model-name to vLLM.",
                file=sys.stderr,
            )

    # ── Validate file arg ──────────────────────────────────
    if not args.file and not agent_config.is_registry_agent:
        parser.error("--file is required for non-registry agents")

    if args.file and not os.path.exists(args.file):
        print(f"[ERROR] File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    # ── Banner ──────────────────────────────────────────────
    print()
    print("=" * 50)
    print("  SOVEREIGN AI WORKBENCH -- ORCHESTRATOR v3")
    print("=" * 50)
    print()
    print(f"  Agent : {agent_config.name if agent_config else 'None'}")
    print(f"  Model : {model_config.name} ({model_config.id})")
    print(f"  Endpoint: {model_config.base_url}")
    if args.image:
        print(f"  Image : {args.image}")
    print()

    # ── Initialize LLM via langchain-openai ────────────────
    llm = ChatOpenAI(
        model=model_config.id,
        base_url=model_config.base_url,
        api_key=model_config.api_key,
        temperature=model_config.temperature,
        max_tokens=model_config.max_tokens,
        timeout=120,
        default_headers={
            "Ngrok-Skip-Browser-Warning": "true",
        },
    )

    # ── Build context ──────────────────────────────────────
    ctx = ContextSchema(
        llm=llm,
        model_config=model_config,
        agent_config=agent_config,
        sandbox_image=sandbox_image,
        sandbox_timeout=sandbox_timeout,
        sandbox_mem_limit=sandbox_mem_limit,
    )

    # ── Initial state ──────────────────────────────────────
    initial_state = {
        "user_prompt": args.prompt,
        "auto": args.auto,
        "extract_attempts": 0,
        "sandbox_attempts": 0,
        "security_attempts": 0,
        "agent_name": agent_config.name if agent_config else "",
    }

    # Set file_path for non-registry agents
    if args.file:
        initial_state["file_path"] = args.file

    # Set image_path for vision agents
    if args.image:
        initial_state["image_path"] = args.image

    # ── Run the graph ──────────────────────────────────────
    try:
        result = app.invoke(initial_state, context=ctx)
    except SystemExit as exc:
        sys.exit(exc.code if isinstance(exc.code, int) else 1)

    # ── Report ──────────────────────────────────────────────
    print()

    sandbox_result = result.get("sandbox_result", {})
    exit_code = sandbox_result.get("exit_code", -1)

    if exit_code == 0:
        print("[SUCCESS] File modified in-place.")
        stdout = sandbox_result.get("stdout", "")
        if stdout:
            print()
            print("--- Output " + "-" * 38)
            print(stdout)
            print("--- End Output " + "-" * 34)
    else:
        print("[FAILED] Sandbox execution error.", file=sys.stderr)
        print(
            f"  Exit code: {exit_code}", file=sys.stderr
        )
        stderr = sandbox_result.get("stderr", "")
        stdout = sandbox_result.get("stdout", "")
        if stderr:
            print()
            print("--- Errors " + "-" * 38)
            print(stderr, file=sys.stderr)
            print("--- End Errors " + "-" * 34)
        if stdout:
            print()
            print("--- Stdout " + "-" * 38)
            print(stdout)
            print("--- End Stdout " + "-" * 34)

    sys.exit(0 if exit_code == 0 else 1)


if __name__ == "__main__":
    main()
