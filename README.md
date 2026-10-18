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

Install the Chromium browser used by the webpage-reading tool:

```powershell
python -m playwright install chromium
```

## Run the project

Start the app from the repository root after installing dependencies and
setting the environment variables:

```powershell
python -m uvicorn app:app --reload --port 8080
```

Open `http://127.0.0.1:8080` in a browser. Conversations are saved in
`database_chatbot.db` and scoped to the browser session. Select either the
local Ollama model `qwen2.5:0.5b` or one of the available Gemini chat models
in the chat header. Gemini 2.5 Flash and Flash-Lite have free-tier access
subject to Google's account eligibility and rate limits. Other Gemini models
may have different availability or pricing. All listed models support chat and
tool calls, and tool activity is shown in the conversation as it happens. The
assistant can read a specific public webpage when asked. Select **Travel
planner** in the sidebar for travel research, airport and flight lookups,
weather forecasts, currency conversion, and budget estimates. Live flight data
requires an `AVIATIONSTACK_API_KEY` in `.env`. To use Qwen, install and start
Ollama, then pull the model:

```powershell
ollama pull qwen2.5:0.5b
```

The model selector in the chat header shows the active model.

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
- `.env.example` — optional API key names for web search and weather
- `LICENSE` — licensing terms

## Environment variables

Copy the example file to create your local configuration:

```powershell
Copy-Item .env.example .env
```

Set `GOOGLE_API_KEY` to enable Gemini. The Tavily key enables web search; the
OpenWeather key is optional and enables weather lookup:

```text
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
```

Keep real credentials in `.env` only. `.env` is ignored by Git; do not commit
it or put real credentials in `.env.example`.
