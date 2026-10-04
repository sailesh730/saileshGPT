# saileshGPT

## Project status

This repository contains the Python dependency list and environment setup
instructions. Application code and its run command/navigation guide have not
been added yet.

## Requirements

- Python 3.10 or newer

## Set up the Python environment

From the repository root, create a virtual environment:

```powershell
py -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
```

On macOS or Linux, create and activate the environment with:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies inside the activated environment:

```powershell
python -m pip install -r requirement.txt
```

## Run the project

The dependencies are prepared for a FastAPI application, but there is no
application entry point in the repository yet. Add the appropriate Uvicorn
command here when the app's entry point is available.

## Navigate the repository

At present, the repository contains:

- `README.md` — setup and project instructions
- `requirement.txt` — Python dependencies
- `.env.example` — names of the API keys the application will need
- `LICENSE` — licensing terms
- `.gitignore` — files and local artifacts excluded from version control

Application folders and navigation instructions can be added when the
application code is introduced.

## Environment variables

Copy the example file to create your local configuration:

```powershell
Copy-Item .env.example .env
```

Set your Google AI and Tavily API keys in `.env`:

```text
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Keep real credentials in `.env` only. `.env` is ignored by Git; do not commit
it or put real credentials in `.env.example`.
