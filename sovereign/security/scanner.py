"""AST + Regex security scanner for LLM-generated Python code.

Runs two layers of analysis:
  1. **AST scan** — parses the code and inspects import statements,
     dangerous function calls, and path literals.
  2. **Regex scan** — pattern-matches for hardcoded secrets, environment
     variable access, and suspicious file-system operations.

Returns a :class:`ScanResult` with a pass/fail verdict, a list of
issues found, a redacted copy of the code, and a human-readable summary.
"""

from __future__ import annotations

import ast
import re
import hashlib
from dataclasses import dataclass, field
from typing import Optional

# ────────────────────────────────────────────────────────────
# DATA CLASSES
# ────────────────────────────────────────────────────────────

@dataclass
class SecurityIssue:
    """A single security finding."""
    severity: str        # "critical" | "warning" | "info"
    category: str        # e.g. "dangerous_import", "secret_leak"
    line: int | None     # 1-indexed line number, or None
    message: str
    suggestion: str


@dataclass
class ScanResult:
    """Aggregated result of a security scan."""
    passed: bool                          # False if any critical issues
    issues: list[SecurityIssue] = field(default_factory=list)
    redacted_code: str = ""
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "issues": [
                {
                    "severity": i.severity,
                    "category": i.category,
                    "line": i.line,
                    "message": i.message,
                    "suggestion": i.suggestion,
                }
                for i in self.issues
            ],
            "summary": self.summary,
        }


# ────────────────────────────────────────────────────────────
# DEFAULT BLOCKED LIST
# ────────────────────────────────────────────────────────────

DEFAULT_BLOCKED_IMPORTS = frozenset({
    "socket", "urllib", "requests", "httpx",
    "subprocess", "ctypes", "shutil", "signal",
    "multiprocessing", "threading", "pickle", "shelve",
    "importlib",
})

DEFAULT_BLOCKED_CALLS = frozenset({
    "eval", "exec", "compile", "__import__",
    "getattr", "setattr", "delattr",
    "globals", "locals", "vars",
})

DEFAULT_BLOCKED_PATTERNS = [
    "eval(", "exec(",
    "__import__(",
    "compile(",
]

# ────────────────────────────────────────────────────────────
# SECRET PATTERNS (regex)
# ────────────────────────────────────────────────────────────

SECRET_PATTERNS: list[tuple[str, str, str]] = [
    # (pattern, category, description)
    (r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\'][A-Za-z0-9_\-]{16,}["\']',
     "secret_leak", "Hardcoded API key"),
    (r'(?i)(secret|token|password|passwd|pwd)\s*[=:]\s*["\'][^\s"\']{8,}["\']',
     "secret_leak", "Hardcoded secret/password"),
    (r'(?i)(aws[_-]?access[_-]?key[_-]?id|aws[_-]?secret[_-]?access[_-]?key)\s*[=:]\s*["\'][A-Za-z0-9/+]{20,}["\']',
     "secret_leak", "Hardcoded AWS key"),
    (r'(?i)bearer\s+[A-Za-z0-9_\-\.]{20,}',
     "secret_leak", "Hardcoded bearer token"),
    (r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----',
     "secret_leak", "Embedded private key"),
]

ENV_ACCESS_PATTERNS: list[tuple[str, str, str]] = [
    (r'os\.environ\s*[.\[]', "env_access", "Direct os.environ access"),
    (r'os\.getenv\s*\(', "env_access", "os.getenv() call"),
    (r'os\.environ\.get\s*\(', "env_access", "os.environ.get() call"),
]

PATH_TRAVERSAL_PATTERNS: list[tuple[str, str, str]] = [
    (r'(?<!\w)\.\.[\\/]', "path_traversal", "Parent directory traversal (../)"),
    (r'["\']\/etc\/', "path_traversal", "Access to /etc/ directory"),
    (r'["\']\/var\/', "path_traversal", "Access to /var/ directory"),
    (r'["\']\/tmp\/', "path_traversal", "Access to /tmp/ directory"),
    (r'os\.path\.join\s*\([^)]*\.\.', "path_traversal", "Path join with traversal"),
]


# ────────────────────────────────────────────────────────────
# PUBLIC API
# ────────────────────────────────────────────────────────────

def scan_code(
    code: str,
    agent_security: dict | None = None,
) -> ScanResult:
    """Run both AST and regex scans on *code*.

    Parameters
    ----------
    code : str
        The Python source code to scan.
    agent_security : dict | None
        Optional agent-specific overrides from the YAML config.
        Keys: ``blocked_imports``, ``blocked_patterns``, ``max_code_lines``.

    Returns
    -------
    ScanResult
        Aggregated scan result with pass/fail and all issues found.
    """
    agent_security = agent_security or {}

    blocked_imports = set(DEFAULT_BLOCKED_IMPORTS)
    if "blocked_imports" in agent_security:
        blocked_imports.update(agent_security["blocked_imports"])

    blocked_patterns = list(DEFAULT_BLOCKED_PATTERNS)
    if "blocked_patterns" in agent_security:
        blocked_patterns.extend(agent_security["blocked_patterns"])

    max_lines = agent_security.get("max_code_lines", 500)

    issues: list[SecurityIssue] = []

    # ── line count check ──────────────────────────────────
    line_count = len(code.strip().splitlines())
    if line_count > max_lines:
        issues.append(SecurityIssue(
            severity="warning",
            category="code_length",
            line=None,
            message=f"Code has {line_count} lines, limit is {max_lines}",
            suggestion="Break code into smaller functions or reduce scope.",
        ))

    # ── AST scan ──────────────────────────────────────────
    issues.extend(_ast_scan(code, blocked_imports))

    # ── regex scan ────────────────────────────────────────
    issues.extend(_regex_scan(code, blocked_patterns))

    # ── redaction ─────────────────────────────────────────
    redacted = _redact_secrets(code)

    # ── verdict ───────────────────────────────────────────
    has_critical = any(i.severity == "critical" for i in issues)
    passed = not has_critical

    summary = _build_summary(issues, passed)

    return ScanResult(
        passed=passed,
        issues=issues,
        redacted_code=redacted,
        summary=summary,
    )


# ────────────────────────────────────────────────────────────
# AST SCAN
# ────────────────────────────────────────────────────────────

def _ast_scan(code: str, blocked_imports: set[str]) -> list[SecurityIssue]:
    issues: list[SecurityIssue] = []

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        issues.append(SecurityIssue(
            severity="critical",
            category="syntax_error",
            line=exc.lineno,
            message=f"Syntax error: {exc.msg}",
            suggestion="Fix the syntax error before proceeding.",
        ))
        return issues

    for node in ast.walk(tree):
        # ── import statements ─────────────────────────────
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split(".")[0]
                if mod in blocked_imports:
                    issues.append(SecurityIssue(
                        severity="critical",
                        category="dangerous_import",
                        line=node.lineno,
                        message=f"Blocked import: {alias.name}",
                        suggestion=f"Remove 'import {alias.name}' — module '{mod}' is not allowed.",
                    ))

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod = node.module.split(".")[0]
                if mod in blocked_imports:
                    issues.append(SecurityIssue(
                        severity="critical",
                        category="dangerous_import",
                        line=node.lineno,
                        message=f"Blocked import: from {node.module}",
                        suggestion=f"Remove 'from {node.module} import ...' — module '{mod}' is not allowed.",
                    ))

        # ── dangerous function calls ──────────────────────
        elif isinstance(node, ast.Call):
            func_name = _get_call_name(node)
            if func_name in DEFAULT_BLOCKED_CALLS:
                issues.append(SecurityIssue(
                    severity="critical",
                    category="dangerous_call",
                    line=node.lineno,
                    message=f"Blocked function call: {func_name}()",
                    suggestion=f"Remove or rewrite the {func_name}() call.",
                ))

        # ── f-strings with path traversal ─────────────────
        elif isinstance(node, ast.JoinedStr):
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    if ".." in value.value or "/etc/" in value.value:
                        issues.append(SecurityIssue(
                            severity="warning",
                            category="path_traversal",
                            line=node.lineno,
                            message="F-string contains path traversal pattern",
                            suggestion="Review the path for directory traversal.",
                        ))

    return issues


def _get_call_name(node: ast.Call) -> str:
    """Extract the function name from a Call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        parts = []
        current = node.func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
    return ""


# ────────────────────────────────────────────────────────────
# REGEX SCAN
# ────────────────────────────────────────────────────────────

def _regex_scan(
    code: str,
    blocked_patterns: list[str],
) -> list[SecurityIssue]:
    issues: list[SecurityIssue] = []
    lines = code.splitlines()

    # ── blocked patterns ──────────────────────────────────
    for pat in blocked_patterns:
        for i, line in enumerate(lines, 1):
            if pat in line:
                issues.append(SecurityIssue(
                    severity="critical",
                    category="blocked_pattern",
                    line=i,
                    message=f"Blocked pattern found: {pat}",
                    suggestion=f"Remove or rewrite the code containing '{pat}'.",
                ))

    # ── secret patterns ───────────────────────────────────
    for pattern, category, desc in SECRET_PATTERNS:
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line):
                issues.append(SecurityIssue(
                    severity="critical",
                    category=category,
                    line=i,
                    message=desc,
                    suggestion="Remove hardcoded secrets. Use environment variables instead.",
                ))

    # ── env access patterns ───────────────────────────────
    for pattern, category, desc in ENV_ACCESS_PATTERNS:
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line):
                issues.append(SecurityIssue(
                    severity="warning",
                    category=category,
                    line=i,
                    message=desc,
                    suggestion="Avoid direct environment access in sandboxed code.",
                ))

    # ── path traversal patterns ───────────────────────────
    for pattern, category, desc in PATH_TRAVERSAL_PATTERNS:
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line):
                issues.append(SecurityIssue(
                    severity="warning",
                    category=category,
                    line=i,
                    message=desc,
                    suggestion="Use relative paths within /workspace/ only.",
                ))

    return issues


# ────────────────────────────────────────────────────────────
# SECRET REDACTION
# ────────────────────────────────────────────────────────────

def _redact_secrets(code: str) -> str:
    """Replace detected secret values with [REDACTED]."""
    redacted = code

    for pattern, _, _ in SECRET_PATTERNS:
        redacted = re.sub(
            pattern,
            lambda m: m.group().split("=")[0].split(":")[0] + "=[REDACTED]"
            if "=" in m.group() or ":" in m.group()
            else "[REDACTED]",
            redacted,
        )

    return redacted


# ────────────────────────────────────────────────────────────
# SUMMARY
# ────────────────────────────────────────────────────────────

def _build_summary(issues: list[SecurityIssue], passed: bool) -> str:
    if not issues:
        return "Security scan passed — no issues found."

    critical = [i for i in issues if i.severity == "critical"]
    warnings = [i for i in issues if i.severity == "warning"]
    info = [i for i in issues if i.severity == "info"]

    parts = []
    if critical:
        parts.append(f"{len(critical)} CRITICAL issue(s) block execution")
    if warnings:
        parts.append(f"{len(warnings)} warning(s)")
    if info:
        parts.append(f"{len(info)} info note(s)")

    verdict = "PASSED (with warnings)" if passed else "BLOCKED"
    return f"Security scan {verdict} — {', '.join(parts)}."
