#!/usr/bin/env python3
"""Sovereign AI Workbench — Backend Orchestrator (v3)

Backward-compatible entry point.  Delegates to the sovereign package.

Usage:
    python orchestrator.py --prompt "Add 5% tax to pricing" --file ./data.xlsx
    python orchestrator.py --prompt "Sort by price" --file ./data.xlsx --auto
    python orchestrator.py --list-agents
    python orchestrator.py --list-models
"""

from sovereign.cli import main

if __name__ == "__main__":
    main()
