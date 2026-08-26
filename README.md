# Sovereign AI Workbench — Backend Orchestrator

A fully air-gapped, local-only AI workbench that uses a local LLM to process files. Zero internet required. Zero data leaves your machines.

---

## Architecture

```
Node 1 (GPU Laptop)                         Node 2 (Mac / Orchestrator)
┌───────────────────────────┐              ┌───────────────────────────────┐
│  vLLM + Qwen2.5-Coder    │              │  orchestrator.py              │
│  6GB VRAM (AWQ quantized)│◄──── WLAN ──►│  + Docker sandbox             │
│                           │              │  + target files               │
└───────────────────────────┘              └───────────────────────────────┘
         │                                          │
         │  No router, no internet                  │  network_mode='none'
         │  Ad-hoc peer-to-peer Wi-Fi               │  Containers cannot phone home
```

**How it works:**

1. You provide a natural-language prompt + a file path on Node 2
2. The script asks the LLM on Node 1 to generate Python code
3. The code is extracted and executed inside an ephemeral, network-disabled Docker container on Node 2
4. The container bind-mounts your file, modifies it in-place, then self-destructs

---

## Prerequisites

| Requirement | Node 1 (GPU) | Node 2 (Mac) |
|---|---|---|
| OS | Linux (recommended) | macOS |
| Python | 3.10+ | 3.10+ |
| RAM | 16 GB | 8 GB+ |
| GPU VRAM | 6 GB+ | N/A |
| Docker | Not required | Docker Desktop |
| Network | Ad-hoc WLAN interface | Ad-hoc WLAN interface |

---

## Step 1: Network Setup (Ad-hoc WLAN)

Both machines must connect to the same ad-hoc Wi-Fi network **before** running anything.

### On Node 1 (GPU Laptop — Linux)

Create an ad-hoc network:

```bash
# Find your wireless interface name
iwconfig

# Create ad-hoc network (replace wlan0 with your interface)
sudo iwconfig wlan0 mode ad-hoc
sudo iwconfig wlan0 essid "SovereignAI" key s:yourpassword
sudo ifconfig wlan0 192.168.1.5 netmask 255.255.255.0 up
```

### On Node 2 (Mac)

Join the same ad-hoc network:

```bash
# System Preferences → Network → Wi-Fi → Join ad-hoc network "SovereignAI"
# Then assign an IP in the same subnet:
sudo ifconfig en0 192.168.1.10 netmask 255.255.255.0 up
```

### Verify Connectivity

```bash
# From Node 2, ping Node 1:
ping 192.168.1.5

# From Node 1, ping Node 2:
ping 192.168.1.10
```

> **Tip:** You can use any IPs in the `192.168.1.x` range as long as both machines are in the same subnet and can ping each other.

---

## Step 2: Node 1 Setup (vLLM + Model)

### Install vLLM

```bash
pip install vllm
```

### Download and Serve the Model

```bash
# Serve the AWQ-quantized Qwen2.5-Coder-7B model
vllm serve Qwen/Qwen2.5-coder-7B-Instruct-AWQ \
    --quantization awq \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096
```

### Verify vLLM is Running

```bash
# From Node 1:
curl http://localhost:8000/v1/models

# From Node 2 (over WLAN):
curl http://192.168.1.5:8000/v1/models
```

You should see a JSON response listing the model name.

---

## Step 3: Node 2 Setup (Orchestrator + Docker)

### Clone / Copy Project Files

Copy the entire project folder to Node 2, or clone it from your source.

### Install Python Dependencies

```bash
cd "Sovereign AI"
pip install -r requirements.txt
```

### Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your actual Node 1 IP:

```ini
VLLM_BASE_URL=http://192.168.1.5:8000/v1
VLLM_MODEL=Qwen2.5-coder-7B-Instruct-AWQ
```

### Build the Sandbox Docker Image

```bash
docker build -t sandbox-python -f Dockerfile.sandbox .
```

This creates a minimal Python image with pandas, openpyxl, and python-docx pre-installed. You only need to build this once.

### Verify Docker is Running

```bash
docker info
docker images sandbox-python
```

---

## Step 4: Running the Orchestrator

### Basic Usage

```bash
python orchestrator.py \
    --prompt "Read vendor_list.xlsx and add a 5% tax to the pricing column" \
    --file "/Users/you/Documents/vendor_list.xlsx"
```

The script will:
1. Validate the file exists
2. Ask the LLM to generate processing code
3. Display the generated code for your review
4. Ask for confirmation before executing
5. Run the code in an isolated container
6. Report success or failure

### Auto Mode (Skip Confirmation)

```bash
python orchestrator.py --auto \
    --prompt "Sort the spreadsheet by price descending" \
    --file "./data.xlsx"
```

### More Examples

```bash
# Merge two sheets
python orchestrator.py \
    --prompt "Merge 'Sheet1' and 'Sheet2' on column 'ID', save to 'merged.xlsx'" \
    --file "./workbook.xlsx"

# Process a Word document
python orchestrator.py \
    --prompt "Extract all headings and bullet points, save as 'outline.txt'" \
    --file "./report.docx"

# Clean CSV data
python orchestrator.py \
    --prompt "Remove duplicate rows, fill missing values with 'N/A', save back" \
    --file "./customers.csv"
```

---

## Security Model

| Layer | Mechanism |
|---|---|
| Network isolation | `network_disabled=True` — containers cannot reach WLAN or internet |
| Filesystem isolation | Only the target file's parent directory is mounted |
| Resource limits | 512 MB RAM, 50% CPU, configurable timeout |
| Ephemeral execution | Container auto-removes after each run |
| No secrets in transit | `api_key` is a dummy string; vLLM does not require authentication |

The generated code **cannot** exfiltrate data, download dependencies, or access the network.

---

## Troubleshooting

### "Cannot connect to Docker daemon"
- Open Docker Desktop and wait for it to fully start
- Check the Docker whale icon is in your menu bar

### "Could not reach vLLM after 3 attempts"
- Verify Node 1 is powered on and running vLLM
- Check that both machines are on the same ad-hoc network
- Run `ping 192.168.1.5` from Node 2 to confirm connectivity
- Verify vLLM is listening on `0.0.0.0:8000` (not just `127.0.0.1`)

### "Sandbox execution failed"
- Check the stderr output — it usually shows the Python error
- Make sure the file format matches what the LLM expects (e.g. `.xlsx` for Excel)
- Try running with a simpler prompt to debug

### "Could not extract valid Python code"
- The LLM may have returned conversational text instead of code
- Try rephrasing your prompt to be more explicit: "Write Python code that..."

### Sandbox timeout
- Increase `SANDBOX_TIMEOUT` in `.env` (default: 60 seconds)
- For large files, the LLM-generated code may need more processing time

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `VLLM_BASE_URL` | `http://192.168.1.5:8000/v1` | Full URL to vLLM's OpenAI-compatible endpoint |
| `VLLM_MODEL` | `Qwen2.5-coder-7B-Instruct-AWQ` | Model identifier (must match vLLM's --model) |
| `SANDBOX_IMAGE` | `sandbox-python:latest` | Docker image for code execution |
| `SANDBOX_TIMEOUT` | `60` | Max seconds for sandbox execution |
| `SANDBOX_MEM_LIMIT` | `512m` | Max RAM for the container |

---

## Project Structure

```
Sovereign AI/
├── orchestrator.py          # Main application — run this
├── Dockerfile.sandbox       # Docker image for isolated code execution
├── requirements.txt         # Python dependencies
├── .env.example             # Configuration template (copy to .env)
└── README.md                # This file
```
