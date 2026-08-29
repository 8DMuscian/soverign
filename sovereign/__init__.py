"""Sovereign AI Workbench — LangGraph-powered multi-agent orchestrator."""

from __future__ import annotations

import os

__version__ = "3.0.1"


def sanitize_ssl_env() -> None:
    """Remove broken SSL/TLS env-var pointers that some httpx builds crash on.

    If ``SSL_CERT_FILE`` or ``CURL_CA_BUNDLE`` points at a file that does not
    exist, httpx (via langchain-openai) raises ``FileNotFoundError`` at client
    construction time.  Drop only the broken entries so a valid custom CA still
    works.
    """
    for env_var in ("SSL_CERT_FILE", "CURL_CA_BUNDLE"):
        val = os.environ.get(env_var)
        if val and not os.path.exists(val):
            del os.environ[env_var]
