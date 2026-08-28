"""Security scanner for LLM-generated code.

Two-layer scanning: AST analysis + Regex pattern matching.
Blocks dangerous imports, secret leaks, env access, and path traversal.
"""
