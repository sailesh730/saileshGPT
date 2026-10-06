# saileshGPT

SaileshGPT is a FastAPI chat assistant with saved conversation history,
incremental response streaming, and document uploads for chat-based search.

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

Start the app from the repository root after installing dependencies and
setting the environment variables:

```powershell
python -m uvicorn app:app --reload --port 8080
```

Open `http://127.0.0.1:8080` in a browser. Conversations are saved in
`database_chatbot.db` and scoped to the browser session. Assistant responses
stream as they are generated. Choose the Gemini model from the selector in the
chat header; the selected model is remembered in the browser and used for
subsequent messages.

Use the attachment button to upload PDF, DOCX, TXT, Markdown, Python, or CSV
documents up to 20 MB. Uploaded documents are indexed for the current
conversation in the local `chroma-db` directory.

## Navigate the repository

The main application files are:

- `app.py` — FastAPI routes for chat, history, and uploads
- `templates/index.html` — chat interface
- `database.py` — conversation and message persistence
- `rag.py` — document extraction and search indexing
- `requirement.txt` — Python dependencies
- `.env.example` — required API key names
- `LICENSE` — licensing terms

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
