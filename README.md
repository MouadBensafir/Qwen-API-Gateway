# Backend

This backend is a standalone FastAPI service that uses a local vLLM OpenAI-compatible endpoint.

## Files

- `config.json`: backend server and vLLM settings
- `app/config.py`: config loading and shared constants
- `app/models.py`: request and response schemas
- `app/session_store.py`: in-memory conversation state
- `app/document_utils.py`: image/PDF extraction helpers
- `app/vllm_client.py`: AsyncOpenAI client for vLLM
- `app/server.py`: FastAPI routes and tool-driven service assistant
- `run.py`: local runner
- `requirements.txt`: Python dependencies
- `templates/`: empty JSON form templates
- `submissions/`: per-session filled JSON outputs

## Install

```bash
cd backend
py -3 -m pip install -r requirements.txt
```

## Run

Start vLLM first, then run:

```bash
cd backend
py -3 run.py
```

Default backend port is `8001`.

## Config

`backend/config.json` supports:

- `server.host`
- `server.port`
- `chat.systemPrompt`
- `chat.assistantStylePrompt`
- `chat.maxToolRounds`
- `chat.recentMessageCount`
- `vllm.baseUrl`
- `vllm.apiKey`
- `vllm.model`
- `vllm.requestTimeoutSeconds`
- `vllm.maxCompletionTokens`
- `vllm.temperature`
- `vllm.pdfVisionMaxPages`
- `vllm.pdfTextMinChars`

## API

Health check:

```bash
curl.exe http://127.0.0.1:8001/health
```

Plain JSON chat:

```bash
curl.exe -X POST http://127.0.0.1:8001/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"I need ID Renewal\"}"
```

Multipart chat with file upload:

```bash
curl.exe -X POST http://127.0.0.1:8001/chat ^
  -F "prompt=I need VISA Appointment" ^
  -F "file=@C:\\path\\to\\passport.jpg"
```

## Behavior

The assistant is tool-driven:

1. Explain available services and help the user choose the right one.
2. Create a per-session JSON submission record in `submissions/`.
3. Use controlled tools to inspect services, read submission state, and update stored fields.
4. Extract grounded details from text, images, and PDFs.
5. Ask focused next-step questions while keeping track of what is already filled.
6. Mark the session complete only after all required fields are present.
