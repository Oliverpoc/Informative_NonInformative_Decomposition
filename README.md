# Informative–Non-Informative Decomposition (aIND)

Approximate Informative–Non-Informative Decomposition for paired data: decompose a field Φ into an informative component Φᴵ and a residual Φᴿ, where Φᴵ explains a target field Ψ⁺ and Φᴿ is irrelevant to it.

## Setup and Run

### 1. Clone the repository

```bash
git clone https://github.com/Oliverpoc/Informative_NonInformative_Decomposition.git
cd Informative_NonInformative_Decomposition
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv aIND_venv
.\aIND_venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python -m venv aIND_venv
source aIND_venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the main script

```bash
python aind_decomposition.py
```

This trains the aIND model on toy data, produces decompositions (Φᴵ, Φᴿ), and writes outputs (plots, metrics) under `results/`.
