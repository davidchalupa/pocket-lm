#!/usr/bin/env python3
"""
Local LLM chat with optional local-document retrieval.

Usage:
    python chat_with_docs.py /path/to/docs_dir

- Ingests .txt and .pdf files from the given directory (non-recursive).
- Splits each file into small chunks, embeds them with SentenceTransformer
  and builds a FAISS index.
- During chat, the system retrieves the top-k relevant chunks and adds them
  as context to the prompt. If no docs dir or no files found, the chat falls
  back to pure local-chat (no retrieval).
"""

import os
import sys
import re
import textwrap
import argparse
import contextlib
from pathlib import Path

# retrieval & embeddings
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # will check later

try:
    import faiss
except Exception:
    faiss = None

import numpy as np
from tqdm.auto import tqdm

# PDF reading
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# local Llama bindings
from llama_cpp import Llama

# config
MODEL_PATH = "models/rocket-3b.Q4_K_M.gguf"  # update if needed
MAX_TOKENS = 150
HISTORY_TURNS = 6
EMBED_BATCH = 16
INDEX_ADD_CHUNK = 512
TOP_K = 3
CHUNK_SIZE = 800   # characters per chunk
CHUNK_OVERLAP = 200
# track how many characters the last progress / retry message printed so we can clear it
LAST_PROGRESS_LEN = 0


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'. Update MODEL_PATH accordingly.")

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Answer the user's questions directly and politely.\n\n"
    "IMPORTANT: When replying, ONLY produce the assistant's reply text. "
    "Do NOT produce 'User:', 'You:', or simulate the user's messages. Stop after the assistant's reply."
)


# sanitization regexes
_SPEAKER_RE = re.compile(r'(^|\n)\s*(user|you|human|client|visitor)\b[:\s]', flags=re.IGNORECASE)
_ASSISTANT_LABEL_RE = re.compile(r'^\s*(assistant|ai|bot)[:\-\s]*', flags=re.IGNORECASE)
_ID_LINE_RE = re.compile(r'^\s*(cmpl-[A-Za-z0-9\-]+|[A-Za-z0-9]{8,}|[A-Za-z0-9\-]{10,})\s*$', flags=re.IGNORECASE)


def sanitize_assistant_text(text, last_user_message=None):
    """Trim speaker labels and remove verbatim repeats and id-like tokens."""
    if not text:
        return ""

    text = _ASSISTANT_LABEL_RE.sub("", text, count=1).strip()

    m = _SPEAKER_RE.search(text)
    if m:
        text = text[:m.start()].strip()

    if last_user_message:
        u = last_user_message.strip()
        if u:
            text = re.sub(r'(?im)^\s*' + re.escape(u) + r'\s*$', '', text, flags=re.MULTILINE).strip()
            if len(u) >= 4:
                text = re.sub(re.escape(u), '', text, flags=re.IGNORECASE).strip()

    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    if len(lines) == 1 and _ID_LINE_RE.match(lines[0]):
        return ""
    filtered_lines = [ln for ln in lines if not _ID_LINE_RE.match(ln)]
    text = "\n".join(filtered_lines).strip()

    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


# instantiate local LLM quietly
print("Loading local LLM (llama.cpp backend)...")
with open(os.devnull, "w") as devnull:
    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        llm = Llama(model_path=MODEL_PATH)


def generate_text(llm, prompt, max_tokens=150):
    """
    Robust generation helper. Uses various llama-cpp-python shapes and suppresses
    native stdout/stderr during calls and iteration.
    Returns raw generated text (may be empty).
    """
    stop_re = re.compile(r'(^|\n)\s*(user|you|human|client|visitor)\b[:\s]', flags=re.IGNORECASE)

    def _stop_index_in(text):
        m = stop_re.search(text)
        return m.start() if m else -1

    def _consume_iterable(resp_iter):
        parts = []
        buffer = ""
        try:
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                for chunk in resp_iter:
                    s = ""
                    if isinstance(chunk, dict):
                        choices = chunk.get("choices") if hasattr(chunk, "get") else None
                        if choices:
                            first = choices[0] if choices else None
                            if isinstance(first, dict):
                                delta = first.get("delta") or {}
                                s = delta.get("content") or delta.get("text") or ""
                                if not s:
                                    s = first.get("text") or ""
                                if not s:
                                    msg = first.get("message")
                                    if isinstance(msg, dict):
                                        s = msg.get("content") or msg.get("text") or ""
                        if not s:
                            _metadata_blacklist = {"id", "object", "model", "created", "type", "hash"}
                            for k, v in chunk.items():
                                if not isinstance(k, str):
                                    continue
                                if k.lower() in _metadata_blacklist:
                                    continue
                                if isinstance(v, str) and v.strip():
                                    s = v
                                    break
                    elif isinstance(chunk, (str, bytes)):
                        s = chunk.decode("utf-8", errors="ignore") if isinstance(chunk, bytes) else chunk
                    else:
                        s = str(chunk)

                    if not s:
                        continue

                    combined = buffer + s
                    stop_idx = _stop_index_in(combined)
                    if stop_idx != -1:
                        return (combined[:stop_idx]).strip()
                    parts.append(s)
                    buffer += s
        except Exception:
            pass
        return "".join(parts).strip()

    # Attempt various API shapes
    try:
        if hasattr(llm, "create_completion"):
            try:
                with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    resp = llm.create_completion(prompt=prompt, max_tokens=max_tokens, stream=True)
                if hasattr(resp, "__iter__") and not isinstance(resp, dict):
                    return _consume_iterable(resp)
                if isinstance(resp, dict):
                    choices = resp.get("choices") or []
                    if choices:
                        first = choices[0]
                        text = (first.get("text") or (first.get("message") or {}).get("content") or "") or ""
                        text = text.strip()
                        stop = _stop_index_in(text)
                        return text[:stop].strip() if stop != -1 else text
            except TypeError:
                pass
            except Exception:
                pass
    except Exception:
        pass

    try:
        if hasattr(llm, "create"):
            try:
                with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    resp = llm.create(prompt=prompt, max_tokens=max_tokens, stream=True)
                if hasattr(resp, "__iter__") and not isinstance(resp, dict):
                    return _consume_iterable(resp)
            except TypeError:
                pass
            except Exception:
                pass
            try:
                with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    resp = llm.create(prompt=prompt, max_tokens=max_tokens)
                if isinstance(resp, dict):
                    choices = resp.get("choices") or []
                    if choices and isinstance(choices[0], dict):
                        text = (choices[0].get("text") or (choices[0].get("message") or {}).get("content") or "") or ""
                        text = text.strip()
                        stop = _stop_index_in(text)
                        return text[:stop].strip() if stop != -1 else text
            except Exception:
                pass
    except Exception:
        pass

    try:
        if hasattr(llm, "create_completion"):
            try:
                with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    resp = llm.create_completion(prompt=prompt, max_tokens=max_tokens, stream=False)
                if isinstance(resp, dict):
                    choices = resp.get("choices") or []
                    if choices:
                        first = choices[0]
                        text = (first.get("text") or (first.get("message") or {}).get("content") or "") or ""
                        text = text.strip()
                        stop = _stop_index_in(text)
                        return text[:stop].strip() if stop != -1 else text
            except Exception:
                pass
    except Exception:
        pass

    try:
        if hasattr(llm, "tokenize") and hasattr(llm, "generate") and hasattr(llm, "detokenize"):
            try:
                prompt_bytes = prompt.encode("utf-8")
                toks_in = llm.tokenize(prompt_bytes)
                parts = []
                buffer = ""
                with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    for t in llm.generate(toks_in, max_tokens=max_tokens):
                        if isinstance(t, int):
                            parts.append(t)
                        else:
                            if isinstance(t, (bytes, bytearray)):
                                s = t.decode("utf-8", errors="ignore")
                            else:
                                s = str(t)
                            buffer += s
                            stop = _stop_index_in(buffer)
                            if stop != -1:
                                return buffer[:stop].strip()
                if parts:
                    try:
                        detok = llm.detokenize(parts)
                        if isinstance(detok, (bytes, bytearray)):
                            detok = detok.decode("utf-8", errors="ignore")
                        text = str(detok).strip()
                        stop = _stop_index_in(text)
                        return text[:stop].strip() if stop != -1 else text
                    except Exception:
                        out = []
                        for tok in parts:
                            try:
                                p = llm.detokenize([tok])
                                if isinstance(p, (bytes, bytearray)):
                                    p = p.decode("utf-8", errors="ignore")
                                out.append(str(p))
                            except Exception:
                                continue
                        text = "".join(out).strip()
                        stop = _stop_index_in(text)
                        return text[:stop].strip() if stop != -1 else text
                stop = _stop_index_in(buffer)
                return buffer[:stop].strip() if stop != -1 else buffer.strip()
            except Exception:
                pass
    except Exception:
        pass

    try:
        if 'resp' in locals() and hasattr(resp, "generations"):
            try:
                text = resp.generations[0][0].text.strip()
                stop = _stop_index_in(text)
                return text[:stop].strip() if stop != -1 else text
            except Exception:
                pass
    except Exception:
        pass

    raise RuntimeError("No compatible generation API found on the Llama object or generation failed.")


def generate_with_auto_extend(llm, prompt, last_user_message=None,
                              initial_max_tokens=MAX_TOKENS, max_attempts=3, increment=150):
    """
    Attempts to get a sanitized non-empty assistant reply. Retries with larger
    token budgets if the sanitized reply is empty or raw output looks id-only/truncated.
    Adds a final fallback generation using a minimal prompt (no retrieved context).
    Returns sanitized reply (or a short fallback message if nothing worked).
    """
    max_tokens = initial_max_tokens
    for attempt in range(1, max_attempts + 1):
        raw = generate_text(llm, prompt, max_tokens=max_tokens)
        sanitized = sanitize_assistant_text(raw, last_user_message)
        if sanitized:
            return sanitized

        raw_str = (raw or "").strip()
        is_id_like = False
        if raw_str:
            lines = [ln for ln in raw_str.splitlines() if ln.strip() != ""]
            if len(lines) == 1 and _ID_LINE_RE.match(lines[0]):
                is_id_like = True
        looks_truncated = raw_str.endswith("...") or raw_str.endswith("..")

        if attempt < max_attempts and (not raw_str or is_id_like or looks_truncated):
            max_tokens += increment
            global LAST_PROGRESS_LEN
            retry_msg = f"Assistant: ... (retrying with token budget {max_tokens})"
            print("\r" + " " * 120 + "\r", end="", flush=True)  # clear a wide area first
            print(retry_msg, end="", flush=True)
            LAST_PROGRESS_LEN = len(retry_msg)
            continue

        # final forced short-answer attempt
        forced_prompt = prompt + "\n\nAssistant: Please answer in one short sentence. If you don't know, reply exactly 'I don't know'."
        try:
            raw_forced = generate_text(llm, forced_prompt, max_tokens=max_tokens + increment)
            sanitized_forced = sanitize_assistant_text(raw_forced, last_user_message)
            if sanitized_forced:
                return sanitized_forced
        except Exception:
            pass

        # If forced attempt failed, do a final NO-CONTEXT attempt (minimal prompt)
        try:
            minimal_prompt = SYSTEM_PROMPT + "\n\nHuman: " + (last_user_message or "").strip() + "\nAssistant:"
            raw_min = generate_text(llm, minimal_prompt, max_tokens=max_tokens + 2 * increment)
            sanitized_min = sanitize_assistant_text(raw_min, last_user_message)
            if sanitized_min:
                return sanitized_min
        except Exception:
            pass

        break

    # final friendly fallback message (instead of the raw "(no assistant reply produced)")
    return "Sorry — I couldn't produce an answer. Please try rephrasing the question or ask something simpler."


# document ingestion / embedding / FAISS
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple sliding-window chunker on characters."""
    if not text:
        return []
    text = text.replace("\r\n", "\n")
    start = 0
    chunks = []
    L = len(text)
    if L <= chunk_size:
        return [text.strip()]
    while start < L:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= L:
            break
        start = end - overlap
    return chunks


def load_files_from_dir(dirpath):
    """
    Load .txt and .pdf files from dirpath (recursive).
    Returns list of dicts: {"id": "<relative_path>#<i>", "text": "<chunk>", "source": "<filename>"}
    Prints debug info about what it finds/reads.
    """
    docs = []
    p = Path(dirpath)
    if not p.exists() or not p.is_dir():
        print(f"[load] docs_dir missing or not a dir: {dirpath}")
        return docs

    # prefer pdfplumber if available for better text extraction
    try:
        import pdfplumber
    except Exception:
        pdfplumber = None

    found = 0
    for f in sorted(p.rglob("*")):  # recursive
        if f.is_dir():
            continue
        lower = f.suffix.lower()
        try:
            if f.stat().st_size == 0:
                print(f"[load] skipping zero-length file: {f}")
                continue
        except Exception as e:
            print(f"[load] could not stat file {f}: {e}")
            continue

        text = ""
        name = str(f.relative_to(p))  # nicer id showing subpath
        if lower == ".txt":
            try:
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
                print(f"[load] read txt: {name} ({len(text)} chars)")
                found += 1
            except Exception as e:
                print(f"[load] failed to read txt {name}: {e}")
                continue

        elif lower == ".pdf":
            if pdfplumber is not None:
                try:
                    with pdfplumber.open(str(f)) as pdf:
                        pages = []
                        for page in pdf.pages:
                            try:
                                pages.append(page.extract_text() or "")
                            except Exception:
                                pages.append("")
                        text = "\n\n".join(pages)
                    print(f"[load] read pdf (pdfplumber): {name} ({len(text)} chars)")
                    found += 1
                except Exception as e:
                    print(f"[load] pdfplumber failed for {name}: {e} -- falling back to PyPDF2")
            if not text:
                # fallback to PyPDF2 if available
                try:
                    import PyPDF2
                    try:
                        reader = PyPDF2.PdfReader(str(f))
                        pages = []
                        for page in reader.pages:
                            try:
                                pages.append(page.extract_text() or "")
                            except Exception:
                                pages.append("")
                        text = "\n\n".join(pages)
                        print(f"[load] read pdf (PyPDF2): {name} ({len(text)} chars)")
                        found += 1
                    except Exception as e:
                        print(f"[load] PyPDF2 failed to read {name}: {e}")
                except Exception:
                    print(f"[load] PyPDF2 not installed; cannot read {name}")

        else:
            # skip other file types
            continue

        if not text or not text.strip():
            print(f"[load] no extractable text from {name}")
            continue

        # split into chunks and add to docs
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            docs.append({"id": f"{name}#{i}", "text": c, "source": name})

    print(f"[load] finished: found {found} source files, produced {len(docs)} chunks")
    return docs



def build_retrieval_index(docs):
    """
    Given docs (list of {"id","text","source"}), compute embeddings and build FAISS index.
    Returns (embed_model, index, docs, embs) where embs is numpy array.
    """
    if not docs:
        return (None, None, docs, None)

    if SentenceTransformer is None:
        raise RuntimeError("SentenceTransformer is required for building embeddings. Install sentence-transformers.")

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [d["text"] for d in docs]
    n = len(texts)

    # batch encode with progress bar
    emb_batches = []
    for i in tqdm(range(0, n, EMBED_BATCH), desc="Embedding", unit="batch"):
        batch = texts[i : i + EMBED_BATCH]
        batch_emb = embed_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        emb_batches.append(batch_emb)
    embs = np.vstack(emb_batches) if len(emb_batches) > 1 else emb_batches[0]
    dim = embs.shape[1]

    # build FAISS index
    if faiss is None:
        raise RuntimeError("faiss is required for retrieval. Install faiss-cpu.")

    index = faiss.IndexFlatL2(dim)
    n_vectors = embs.shape[0]
    for i in tqdm(range(0, n_vectors, INDEX_ADD_CHUNK), desc="Indexing", unit="vecs"):
        index.add(embs[i : i + INDEX_ADD_CHUNK])

    return (embed_model, index, docs, embs)


def retrieve_docs(query, embed_model, index, docs, top_k=TOP_K):
    """Return top_k docs (list of dicts) for the query. Empty list if no index."""
    if index is None or embed_model is None or not docs:
        return []
    q_emb = embed_model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    D, I = index.search(q_emb, k=min(top_k, len(docs)))
    return [docs[i] for i in I[0]]


def build_prompt_with_context(context_chunks, conversation_history, user_message):
    """
    Build prompt including system prompt, small context (retrieved), and conversation history.
    """
    history = conversation_history[-HISTORY_TURNS * 2 :]
    parts = [SYSTEM_PROMPT, "\nContext:"]
    if context_chunks:
        for c in context_chunks:
            parts.append(f"[{c['source']}|{c['id']}] {c['text']}")
    else:
        parts.append("(no relevant context found)")

    parts.append("\nConversation:")
    for item in history:
        role_label = "Human" if item["role"] == "user" else "Assistant"
        parts.append(f"{role_label}: {item['text']}")
    parts.append(f"Human: {user_message}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def chat_loop(embed_model=None, index=None, docs=None):
    global LAST_PROGRESS_LEN
    print("\n=== Local Chat (with optional document retrieval) ===")
    print("Type 'exit' or 'quit' to stop.\n")
    history = []
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye.")
                break

            # retrieve context if available
            retrieved = retrieve_docs(user_input, embed_model, index, docs, top_k=TOP_K)
            prompt = build_prompt_with_context(retrieved, history, user_input)

            display_prefix = "Assistant: ..."
            print(display_prefix, end='', flush=True)
            # do not overwrite a possibly-larger progress length set by retries
            LAST_PROGRESS_LEN = max(LAST_PROGRESS_LEN, len(display_prefix))

            # get sanitized reply via auto-extend wrapper
            reply = generate_with_auto_extend(llm, prompt, last_user_message=user_input,
                                              initial_max_tokens=MAX_TOKENS, max_attempts=3, increment=150)
            if not reply:
                reply = "(no assistant reply produced)"

            full_line = "Assistant: " + reply
            # ensure we clear enough characters to cover the longest progress/retry message
            clear_width = max(len(display_prefix), len(full_line), LAST_PROGRESS_LEN)
            # clear line and carriage return, then print full reply
            print('\r' + ' ' * clear_width + '\r', end='', flush=True)
            print(full_line)
            print()  # blank line before next prompt

            # reset progress tracker
            LAST_PROGRESS_LEN = 0

            history.append({"role": "user", "text": user_input})
            history.append({"role": "assistant", "text": reply})
    except KeyboardInterrupt:
        print("\nInterrupted. Bye.")


def main():
    ap = argparse.ArgumentParser(description="Local LLM chat with optional local-doc retrieval (PDF/txt).")
    ap.add_argument("docs_dir", nargs="?", default=None, help="Directory containing .txt/.pdf files to ingest (optional).")
    args = ap.parse_args()

    embed_model = None
    index = None
    docs = []

    if args.docs_dir:
        docs_dir = args.docs_dir
        print(f"Loading files from: {docs_dir}")
        docs = load_files_from_dir(docs_dir)
        print(f"Found {len(docs)} text chunks from files.")
        if docs:
            try:
                embed_model, index, docs, embs = build_retrieval_index(docs)
                print("Built retrieval index.")
            except Exception as e:
                print("Warning: failed to build retrieval index:", e)
                embed_model = None
                index = None

    # start chat loop (passes None for embed_model/index if not available)
    chat_loop(embed_model=embed_model, index=index, docs=docs)


if __name__ == "__main__":
    main()
