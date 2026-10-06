from dotenv import load_dotenv
import os
import certifi
import re
import secrets
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()  # fixed: was REQUEST_CA_BUNDLE (typo, ignored by requests)

import uuid

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from langchain_core.messages import AIMessageChunk, HumanMessage

from database import (
    SessionLocal,
    create_or_update_conversation,
    get_chat_history,
    get_user_conversation,
    get_user_conversations,
    init_db,
    save_chat_message,
)
from model_config import DEFAULT_MODEL, MODEL_OPTIONS, MODEL_IDS

logger = logging.getLogger("saileshgpt")

SESSION_COOKIE = "sailesh_session"
SESSION_PATTERN = re.compile(r"[A-Za-z0-9_-]{40,50}")
ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".py", ".csv"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_MESSAGE_CHARS = 10_000

Path("upload").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # replaces the deprecated @app.on_event("startup")
    init_db()
    yield


app = FastAPI(title="SaileshGPT", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


def get_browser_session_id(request: Request):
    session_id = request.cookies.get(SESSION_COOKIE, "")
    if not SESSION_PATTERN.fullmatch(session_id):
        raise HTTPException(status_code=401, detail="Refresh the page to start a session.")
    return session_id


@app.get("/models")
async def list_models():
    return JSONResponse(
        {
            "default_model": DEFAULT_MODEL,
            "models": [
                {"id": model_id, "label": label}
                for model_id, label in MODEL_OPTIONS
            ],
        }
    )


@app.get("/")
async def home(request: Request):
    response = templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )
    if not SESSION_PATTERN.fullmatch(request.cookies.get(SESSION_COOKIE, "")):
        response.set_cookie(
            key=SESSION_COOKIE,
            value=secrets.token_urlsafe(32),
            max_age=60 * 60 * 24 * 365,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="lax",
        )
    return response


# Plain `def` endpoints run in FastAPI's threadpool, so blocking DB calls
# no longer freeze the event loop.
@app.get("/conversations")
def list_conversations(request: Request):
    session_id = get_browser_session_id(request)
    db = SessionLocal()
    try:
        conversations = get_user_conversations(db, session_id)
        return JSONResponse(
            [
                {
                    "thread_id": conversation.thread_id,
                    "title": conversation.title or "New conversation",
                    "updated_at": conversation.updated_at.isoformat(),
                }
                for conversation in conversations
            ]
        )
    finally:
        db.close()


@app.get("/conversations/{thread_id}")
def load_conversation(thread_id: str, request: Request):
    session_id = get_browser_session_id(request)
    db = SessionLocal()
    try:
        conversation = get_user_conversation(db, thread_id, session_id)
        if conversation is None:
            return JSONResponse({"error": "Conversation not found."}, status_code=404)

        messages = get_chat_history(db, thread_id)
        return JSONResponse(
            {
                "thread_id": conversation.thread_id,
                "title": conversation.title or "New conversation",
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in messages
                ],
            }
        )
    finally:
        db.close()


def thread_belongs_to_session(thread_id: str, session_id: str) -> bool:
    db = SessionLocal()
    try:
        return get_user_conversation(db, thread_id, session_id) is not None
    finally:
        db.close()


async def prepare_chat_request(request: Request):
    session_id = get_browser_session_id(request)
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body."}, status_code=400)

    if not isinstance(payload, dict):
        return JSONResponse({"error": "Invalid JSON body."}, status_code=400)

    message = payload.get("message")
    message = message.strip() if isinstance(message, str) else ""
    if not message:
        return JSONResponse({"error": "Message is required."}, status_code=400)
    if len(message) > MAX_MESSAGE_CHARS:
        return JSONResponse(
            {"error": f"Message is too long (max {MAX_MESSAGE_CHARS} characters)."},
            status_code=413,
        )

    thread_id = payload.get("thread_id")
    if thread_id is not None and not isinstance(thread_id, str):
        return JSONResponse({"error": "Invalid conversation ID."}, status_code=400)

    is_new = not thread_id
    if thread_id:
        if not await run_in_threadpool(thread_belongs_to_session, thread_id, session_id):
            return JSONResponse({"error": "Conversation not found."}, status_code=404)
    else:
        thread_id = f"chat_{uuid.uuid4().hex}"

    model_id = payload.get("model_id", DEFAULT_MODEL)
    if not isinstance(model_id, str) or model_id not in MODEL_IDS:
        return JSONResponse({"error": "Select a supported model."}, status_code=400)

    return session_id, message, thread_id, is_new, model_id


def get_reply_chunks(message: str, thread_id: str, model_id: str):
    from agent import get_agent, get_retry_model_candidates, is_retryable_model_error

    last_error = None
    for candidate_model in get_retry_model_candidates(model_id):
        try:
            agent = get_agent(candidate_model)
            config = {"configurable": {"thread_id": thread_id}}
            for chunk, _ in agent.stream(
                {"messages": [HumanMessage(content=message)]},
                config=config,
                stream_mode="messages",
            ):
                # fixed: "messages" mode also emits ToolMessages and other node output;
                # only the model's own tokens belong in the reply.
                if not isinstance(chunk, AIMessageChunk):
                    continue

                content = chunk.content
                if isinstance(content, str):
                    if content:
                        yield content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, str):
                            yield item
                        elif (
                            isinstance(item, dict)
                            and item.get("type", "text") == "text"
                            and isinstance(item.get("text"), str)
                        ):
                            yield item["text"]
            return
        except Exception as exc:
            last_error = exc
            if not is_retryable_model_error(exc):
                raise

    raise RuntimeError(
        "The selected Gemini model is temporarily unavailable. Please try again in a moment or switch to a different model."
    ) from last_error


def save_chat_turn(session_id: str, thread_id: str, message: str, reply: str, is_new: bool):
    db = SessionLocal()
    try:
        conversation = create_or_update_conversation(
            db,
            thread_id=thread_id,
            user_id=session_id,
            title=message[:80] if is_new else None,
        )
        save_chat_message(db, thread_id, "user", message)
        save_chat_message(db, thread_id, "assistant", reply)
        return conversation.title or "New conversation"
    finally:
        db.close()


def run_chat(session_id: str, message: str, thread_id: str, is_new: bool, model_id: str):
    reply = "".join(get_reply_chunks(message, thread_id, model_id)).strip()
    reply = reply or "I did not receive a response."
    title = save_chat_turn(session_id, thread_id, message, reply, is_new)
    return reply, title


@app.post("/chat")
async def chat(request: Request):
    prepared = await prepare_chat_request(request)
    if isinstance(prepared, JSONResponse):
        return prepared
    session_id, message, thread_id, is_new, model_id = prepared

    try:
        # fixed: the agent call is blocking, so run it off the event loop
        reply, title = await run_in_threadpool(
            run_chat, session_id, message, thread_id, is_new, model_id
        )
        return JSONResponse({"reply": reply, "thread_id": thread_id, "title": title})
    except Exception as exc:
        logger.exception("Chat request failed")
        return JSONResponse(
            {"error": str(exc) or "Chat request failed. Check the server logs for details."},
            status_code=500,
        )


@app.post("/chat/stream")
async def chat_stream(request: Request):
    prepared = await prepare_chat_request(request)
    if isinstance(prepared, JSONResponse):
        return prepared
    session_id, message, thread_id, is_new, model_id = prepared

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    # fixed: a sync generator is iterated in a worker thread by Starlette,
    # so one streaming reply no longer blocks every other request.
    def event_stream():
        reply_parts = []
        try:
            for text in get_reply_chunks(message, thread_id, model_id):
                reply_parts.append(text)
                yield sse({"type": "delta", "text": text})

            reply = "".join(reply_parts).strip() or "I did not receive a response."
            title = save_chat_turn(session_id, thread_id, message, reply, is_new)
            yield sse(
                {
                    "type": "complete",
                    "thread_id": thread_id,
                    "title": title,
                    "reply": reply,
                    "model_id": model_id,
                }
            )
        except Exception as exc:
            logger.exception("Streaming chat request failed")
            yield sse(
                {
                    "type": "error",
                    "message": str(exc) or "Chat request failed. Check the server logs for details.",
                }
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def get_or_create_upload_thread(thread_id, session_id, original_name):
    """Returns (thread_id, title) or None if the thread isn't this session's."""
    if not thread_id:
        return f"chat_{uuid.uuid4().hex}", original_name[:80]
    db = SessionLocal()
    try:
        conversation = get_user_conversation(db, thread_id, session_id)
        if conversation is None:
            return None
        return thread_id, conversation.title or original_name[:80]
    finally:
        db.close()


def save_upload_messages(session_id, thread_id, title, original_name, confirmation):
    db = SessionLocal()
    try:
        create_or_update_conversation(
            db, thread_id=thread_id, user_id=session_id, title=title
        )
        save_chat_message(db, thread_id, "user", f"Uploaded file: {original_name}")
        save_chat_message(db, thread_id, "assistant", confirmation)
    finally:
        db.close()


@app.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    thread_id: str | None = Form(default=None),
):
    from rag import add_document_to_rag

    session_id = get_browser_session_id(request)
    original_name = Path((file.filename or "").replace("\\", "/")).name
    extension = Path(original_name).suffix.lower()
    if not original_name or extension not in ALLOWED_UPLOAD_EXTENSIONS:
        await file.close()
        return JSONResponse(
            {"error": "Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV files."},
            status_code=400,
        )

    thread = await run_in_threadpool(
        get_or_create_upload_thread, thread_id, session_id, original_name
    )
    if thread is None:
        await file.close()
        return JSONResponse({"error": "Conversation not found."}, status_code=404)
    thread_id, title = thread

    stored_path = Path("upload") / f"{uuid.uuid4().hex}{extension}"
    total_bytes = 0
    is_too_large = False
    try:
        with stored_path.open("xb") as destination:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    is_too_large = True
                    break
                destination.write(chunk)

        if is_too_large:
            stored_path.unlink(missing_ok=True)
            return JSONResponse(
                {"error": "File is too large. Maximum upload size is 20 MB."},
                status_code=413,
            )

        # fixed: parsing/embedding a document is slow and blocking
        result = await run_in_threadpool(
            add_document_to_rag, str(stored_path), thread_id, source_name=original_name
        )
        confirmation = f"Uploaded and indexed {original_name} ({result['chunks']} text chunks)."

        await run_in_threadpool(
            save_upload_messages, session_id, thread_id, title, original_name, confirmation
        )

        return JSONResponse(
            {
                "thread_id": thread_id,
                "title": title,
                "filename": original_name,
                "chunks": result["chunks"],
                "message": confirmation,
            }
        )
    except ValueError as exc:
        stored_path.unlink(missing_ok=True)
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception:
        stored_path.unlink(missing_ok=True)
        logger.exception("Upload failed")
        return JSONResponse(
            {"error": "Upload failed. Check the server logs for details."},
            status_code=500,
        )
    finally:
        await file.close()


if __name__ == "__main__":
    # 127.0.0.1 keeps the dev server private; use 0.0.0.0 only when you
    # deliberately want other devices on your network to reach it.
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)