# Sovereign AI Workbench — LangGraph Multi-Agent Orchestrator

A fully air-gapped, local-only AI workbench that uses local LLMs to process files, images, and documents. Built with **LangGraph** and **LangChain** for stateful, graph-based orchestration with automatic retry loops, security scanning, and YAML-based agent/model management. Zero internet required. Zero data leaves your machines.

---

## Architecture

```
Node 1 (GPU Laptop)                         Node 2 (Mac / Orchestrator)
┌───────────────────────────┐              ┌───────────────────────────────┐
│  vLLM + LLM models        │              │  sovereign/graph.py           │
│  (coder + vision models)  │◄──── WLAN ──►│  + Docker sandbox             │
│                           │              │  + Streamlit frontend         │
└───────────────────────────┘              │  + Security scanner           │
         │                                 │  + YAML agent/model registry  │
         │  No router, no internet         └───────────────────────────────┘
         │  Ad-hoc peer-to-peer Wi-Fi               │
         │                                          │  network_mode='none'
         │                                          │  Containers cannot phone home
```

**How it works:**

1. You provide a natural-language prompt + a file path (or use the Model Registry agent)
2. A LangGraph `StateGraph` orchestrates: validate → prompt → LLM → extract → security scan → confirm → sandbox
3. Security scanning blocks dangerous code before execution
4. YAML-based agent and model registries allow zero-code agent/model management
5. Code runs inside ephemeral, network-disabled Docker containers

---

## Graph Flow

```
START
  │
  ▼
validate_file ──► build_prompt ──► call_llm ──► extract_code
                                                │
                                     security_scan
                                                │
                                   ┌────────────┼────────────┐
                                   │            │            │
                              [clean]     [critical]    [retry]
                                   │            │            │
                                   ▼            ▼            ▼
                                confirm    inject_      call_llm
                                   │      security
                                   │       error
                             ┌─────┼─────┐
                        [approved]  [rejected]
                             │          │
                             ▼          ▼
                        run_sandbox    END
                        or write_yaml
                             │
                       [success] │ [fail + retry]
                           │     │        │
                           ▼     ▼        │
                         END  inject_error│
                               │          │
                               └──► call_llm
```

---

## Project Structure

```
Sovereign AI/
├── sovereign/
│   ├── __init__.py              # Package init (v3.0.0)
│   ├── state.py                 # TypedDict graph state definition
│   ├── nodes.py                 # Graph node functions
│   ├── graph.py                 # LangGraph StateGraph with retry + security edges
│   ├── cli.py                   # CLI entry point + registries
│   ├── vision.py                # Multimodal message builder for vision models
│   ├── models/                  # YAML model registry
│   │   ├── __init__.py          # ModelRegistry + ModelConfig
│   │   ├── qwen-coder-7b.yaml  # Coder model config
│   │   └── qwen-vl-7b.yaml     # Vision model config
│   ├── agents/                  # YAML agent registry
│   │   ├── __init__.py          # AgentRegistry + AgentConfig
│   │   ├── data_processing.yaml
│   │   ├── image_processing.yaml
│   │   ├── document_processing.yaml
│   │   └── model_registry.yaml  # Agent that manages models via YAML
│   ├── security/                # Code security scanner
│   │   ├── __init__.py
│   │   └── scanner.py           # AST + Regex analysis
│   └── frontend/
│       ├── __init__.py
│       └── app.py               # Streamlit dashboard
├── orchestrator.py              # Backward-compatible entry point
├── Dockerfile.sandbox           # Docker image for isolated code execution
├── requirements.txt             # Python dependencies
├── .env.example                 # Configuration template
└── README.md                    # This file
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd "Sovereign AI"
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Model and agent configuration is now in YAML files:
- **Models:** `sovereign/models/*.yaml`
- **Agents:** `sovereign/agents/*.yaml`

### 3. Build Sandbox Docker Image

```bash
docker build -t sandbox-python -f Dockerfile.sandbox .
```

### 3b. Start vLLM (WSL on this box)

The coder model is served from the `Ubuntu` WSL distro on this machine using a
venv named `vllm-env`. Launch it (stays in the foreground):

```bash
wsl
source ~/vllm-env/bin/activate
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WSL2_ENABLE_PIN_MEMORY=1 vllm serve \
    Qwen/Qwen2.5-Coder-3B-Instruct-AWQ \
    --quantization awq \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager
```

Verify from Windows: `curl http://localhost:8000/v1/models` should list
`Qwen/Qwen2.5-Coder-3B-Instruct-AWQ`. (The orchestrator and frontend call vLLM
via `localhost`; if you ever move vLLM to another machine, change `base_url` in
`sovereign/models/*.yaml`.)

### 4. Run the Orchestrator

```bash
# List available agents
python orchestrator.py --list-agents

# List registered models
python orchestrator.py --list-models

# Process a file
python orchestrator.py \
    --agent "Data Processor" \
    --prompt "Add 5% tax to pricing column" \
    --file ./data.xlsx

# Image processing (requires vision model)
python orchestrator.py \
    --agent "Image Processor" \
    --prompt "Describe this image" \
    --file ./photo.jpg \
    --image ./photo.jpg

# Register a new model (no code changes needed)
python orchestrator.py \
    --agent "Model Registry" \
    --prompt "Add Qwen3-coder-8B, endpoint http://192.168.0.116:8000/v1, make it default"
```

### 5. Run the Frontend

```bash
streamlit run sovereign/frontend/app.py --server.port 8501
```

Open `http://localhost:8501` in your browser.

The Work tab runs the full pipeline. After generating you get:

- **Execution Result panel** — status banner (success/failure + exit code), sandbox
  stdout, and errors
- **Download modified file** — get the processed file straight from the browser
- **Save copy to folder** — optionally write the result to a real folder on Node 2
  (e.g. your data directory) so you can open it in Excel
- **Persistent workspace** — uploads are staged in `sovereign_frontend_tmp/`; each
  new prompt keeps editing the *already-modified* copy, so multi-step edits work
  like the CLI (use **Reset workspace** to upload fresh originals)

---

## YAML Agent Registry

Add, remove, or replace agents by editing YAML files in `sovereign/agents/`. Zero Python code changes required.

### Agent YAML Schema

```yaml
name: "Data Processor"
description: "Process structured data (Excel, CSV, JSON)"
category: "data_processing"
version: "1.0.0"
enabled: true
icon: "📊"

supported_file_types:
  - ".xlsx"
  - ".csv"
  - ".json"

requires_vision: false
sandbox_image: "sandbox-python:latest"
model: "Qwen Coder 3B"  # optional — uses default if omitted

system_prompt: |
  You are a Python data-processing agent...

security:
  blocked_imports: ["socket", "urllib", "requests"]
  blocked_patterns: ["eval(", "exec("]
  max_code_lines: 300
```

### Adding a New Agent

1. Create a new `.yaml` file in `sovereign/agents/`
2. Restart the orchestrator — agent appears in `--list-agents`

### Replacing an Agent

1. Edit the existing `.yaml` file
2. Restart — changes take effect

### Disabling an Agent

Set `enabled: false` in the YAML file.

---

## YAML Model Registry

Add, remove, or replace LLM models by editing YAML files in `sovereign/models/`. Zero code changes required.

### Model YAML Schema

```yaml
name: "Qwen Coder 3B"
id: "Qwen/Qwen2.5-Coder-3B-Instruct-AWQ" # must match what vLLM serves
description: "Fast code generation for data/document processing"
base_url: "http://localhost:8000/v1" # vLLM endpoint (runs in WSL on this box)
api_key: "not-needed"
capabilities:
  - code_generation
  - text_generation
requires_vision: false
default: true
max_tokens: 2048                        # per-model generation headroom
temperature: 0.1
max_model_len: 4096                     # align with vLLM --max-model-len
```

> **Auto-matching:** on startup the system queries `/v1/models` on the configured
> `base_url`. If the `id` doesn't match exactly, it auto-resolves to the closest
> served model (e.g. a short id matching `Qwen/Qwen2.5-Coder-3B-Instruct-AWQ`)
> and warns you. Keeping the full repo path (`Qwen/...`) as `id` is safest.

### Adding a New Model

1. Create a new `.yaml` file in `sovereign/models/`
2. Restart — model appears in `--list-models`

Or use the Model Registry agent to add it automatically:

```bash
python orchestrator.py \
    --agent "Model Registry" \
    --prompt "Add Qwen3-coder-8B, endpoint http://192.168.0.116:8000/v1, make it default"
```

---

## Security Scanner

Every generated code block is scanned before execution:

| Layer | What it catches |
|---|---|
| **AST scan** | Dangerous imports (`socket`, `subprocess`, `ctypes`), dangerous calls (`eval`, `exec`, `compile`) |
| **Regex scan** | Hardcoded secrets (API keys, tokens, passwords), environment access (`os.environ`), path traversal (`../`) |
| **Agent-specific** | Each agent YAML can add extra `blocked_imports` and `blocked_patterns` |

Critical issues block execution and loop back to the LLM with feedback. Warnings are displayed but don't block.

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `SANDBOX_IMAGE` | `sandbox-python:latest` | Docker image for code execution |
| `SANDBOX_TIMEOUT` | `60` | Max seconds for sandbox execution |
| `SANDBOX_MEM_LIMIT` | `512m` | Max RAM for the container |
| `MAX_EXTRACT_RETRIES` | `3` | Max code extraction retry attempts |
| `MAX_SANDBOX_RETRIES` | `2` | Max sandbox execution retry attempts |
| `MAX_SECURITY_RETRIES` | `2` | Max security scan retry attempts |
| `FRONTEND_PORT` | `8501` | Streamlit frontend port |

---

## Tech Stack

| Component | Library | Purpose |
|---|---|---|
| Graph orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) | Stateful graph with nodes, conditional edges, retry loops |
| LLM integration | [LangChain](https://github.com/langchain-ai/langchain) + [langchain-openai](https://github.com/langchain-ai/langchain/tree/master/libs/partners/openai) | `ChatOpenAI` pointed at vLLM's OpenAI-compatible API |
| State management | `TypedDict` + `add_messages` reducer | Type-safe state with auto-merging message history |
| Security | AST + Regex scanner | Two-layer code analysis before execution |
| Configuration | PyYAML + `python-dotenv` | YAML agent/model configs, `.env` for runtime settings |
| Frontend | [Streamlit](https://streamlit.io) | Multi-page dashboard for agent selection, code review, security |
| Sandbox | Docker SDK for Python | Isolated, network-disabled code execution |

---

## Troubleshooting

### "Cannot connect to Docker daemon"
- Open Docker Desktop and wait for it to fully start
- Check the Docker whale icon is in your menu bar

### "Could not reach vLLM"
- Verify Node 1 is powered on and running vLLM
- Check that both machines are on the same ad-hoc network
- Run the vLLM WSL command in section 3b, then verify with
  `curl http://localhost:8000/v1/models`

### "Model X does not exist (404 from vLLM)"
- The `id` in the model YAML does not match what vLLM serves
- Confirm with `curl http://localhost:8000/v1/models` and set the `id` to the
  exact served name (this repo uses the full repo path `Qwen/...`)
- Since v3.0 the system auto-matches to the closest served name and warns you

### "FileNotFoundError / SSL_CERT_FILE on startup"
- A broken `SSL_CERT_FILE` or `CURL_CA_BUNDLE` env var crashes httpx
- The system now sanitizes broken pointers automatically; unset them if the
  warning persists

### "Agent not found"
- Run `python orchestrator.py --list-agents` to see available agents
- Check the agent name matches exactly (case-insensitive)

### "Model not found"
- Run `python orchestrator.py --list-models` to see registered models
- Check `sovereign/models/` directory for YAML files

### Security scan blocks my code
- Review the security issues in the scan output
- The LLM will be asked to regenerate code avoiding the flagged patterns
- Adjust `blocked_imports` / `blocked_patterns` in the agent YAML if needed

### Sandbox timeout
- Increase `SANDBOX_TIMEOUT` in `.env` (default: 60 seconds)
- For large files, the LLM-generated code may need more processing time
