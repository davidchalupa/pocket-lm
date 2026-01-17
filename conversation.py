"""
Minimal local LLM chat that displays:
  Assistant: ...      (while model is generating)
and then replaces that with:
  Assistant: <reply>
"""

import os
import re
import textwrap
import contextlib
from llama_cpp import Llama


# config
MODEL_PATH = "models/rocket-3b.Q4_K_M.gguf"  # <- update to your model file
MAX_TOKENS = 150
HISTORY_TURNS = 6   # how many past user + assistant turns to include


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'. Update MODEL_PATH accordingly.")


SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Answer the user's questions directly and politely.\n\n"
    "IMPORTANT: When replying, ONLY produce the assistant's reply text. "
    "Do NOT produce 'User:', 'You:', or simulate the user's messages. Stop after the assistant's reply."
)


# regexes for sanitization / trimming
_SPEAKER_RE = re.compile(r'(^|\n)\s*(user|you|human|client|visitor)\b[:\s]', flags=re.IGNORECASE)
_ASSISTANT_LABEL_RE = re.compile(r'^\s*(assistant|ai|bot)[:\-\s]*', flags=re.IGNORECASE)
_ID_LINE_RE = re.compile(r'^\s*(cmpl-[A-Za-z0-9\-]+|[A-Za-z0-9]{8,}|[A-Za-z0-9\-]{10,})\s*$', flags=re.IGNORECASE)


def sanitize_assistant_text(text, last_user_message=None):
    """Trim speaker labels and remove verbatim repeats of the user's message.
    Also remove single-line id-like tokens (e.g. 'cmpl-...') that are not helpful.
    """
    if not text:
        return ""

    # Remove leading assistant label if present
    text = _ASSISTANT_LABEL_RE.sub("", text, count=1).strip()

    # Trim anything from the first user-like speaker label onward
    m = _SPEAKER_RE.search(text)
    if m:
        text = text[:m.start()].strip()

    # Remove exact-line repetitions of the user's last message
    if last_user_message:
        u = last_user_message.strip()
        if u:
            text = re.sub(r'(?im)^\s*' + re.escape(u) + r'\s*$', '', text, flags=re.MULTILINE).strip()
            if len(u) >= 4:
                text = re.sub(re.escape(u), '', text, flags=re.IGNORECASE).strip()

    # Remove single-line id-like outputs (e.g., cmpl-..., or short alphanum tokens)
    # If whole reply is just an id-like token, return empty string
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    if len(lines) == 1 and _ID_LINE_RE.match(lines[0]):
        return ""

    # Also remove any id-like lines embedded inside the reply
    filtered_lines = [ln for ln in lines if not _ID_LINE_RE.match(ln)]
    text = "\n".join(filtered_lines).strip()

    # Normalize whitespace
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


print("Loading local LLM (llama.cpp backend)...")
# instantiate the local model while suppressing stdout/stderr from the native backend
with open(os.devnull, "w") as devnull:
    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        llm = Llama(model_path=MODEL_PATH)


def generate_text(llm, prompt, max_tokens=150):
    """
    Same behavior as before, but suppresses native-backend stdout (and stderr)
    during llama-cpp calls and iteration so backend lines like
    "Llama.generate: 92 prefix-match hit..." are not printed.
    """
    stop_re = re.compile(r'(^|\n)\s*(user|you|human|client|visitor)\b[:\s]', flags=re.IGNORECASE)

    def _stop_index_in(text):
        m = stop_re.search(text)
        return m.start() if m else -1

    def _consume_iterable(resp_iter):
        parts = []
        buffer = ""
        try:
            # iterate while suppressing native stdout/stderr
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
                            # safer fallback: iterate items, skip metadata keys that often contain ids
                            _metadata_blacklist = {"id", "object", "model", "created", "type", "hash"}
                            for k, v in chunk.items():
                                if not isinstance(k, str):
                                    continue
                                kl = k.lower()
                                if kl in _metadata_blacklist:
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
            # fall back to whatever was collected
            pass
        return "".join(parts).strip()

    # 1) try modern create_completion with stream True (collect)
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

    # 2) try older .create() streaming or non-stream
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

    # 3) try non-streaming create_completion (if present)
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

    # 4) low-level fallback: tokenize -> generate -> detokenize (collect tokens)
    try:
        if hasattr(llm, "tokenize") and hasattr(llm, "generate") and hasattr(llm, "detokenize"):
            try:
                prompt_bytes = prompt.encode("utf-8")
                toks_in = llm.tokenize(prompt_bytes)
                parts = []
                buffer = ""
                # iterate while suppressing native stdout/stderr
                with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    for t in llm.generate(toks_in, max_tokens=max_tokens):
                        # if token is int, collect; otherwise try to coerce to string
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

    # 5) fallback: resp.generations
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
    Attempts to get a *sanitized* non-empty assistant reply. Retries with larger
    token budgets if the sanitized reply is empty or the raw output looks like an id-only result.

    Returns the sanitized assistant reply (string). If all attempts fail, returns an empty string.
    """
    max_tokens = initial_max_tokens
    last_raw = ""
    for attempt in range(1, max_attempts + 1):
        # generate raw output (uses your robust generate_text)
        raw = generate_text(llm, prompt, max_tokens=max_tokens)
        last_raw = raw or ""

        # sanitize it (same sanitizer you already have)
        sanitized = sanitize_assistant_text(last_raw, last_user_message)

        # If sanitized reply is non-empty, accept it
        if sanitized:
            return sanitized

        # If sanitized empty, decide to retry based on heuristics:
        # - raw is empty
        # - raw is id-like (single-line cmpl-...)
        # - raw ends with ellipses suggesting truncation
        raw_str = (last_raw or "").strip()
        is_id_like = False
        if raw_str:
            lines = [ln for ln in raw_str.splitlines() if ln.strip() != ""]
            if len(lines) == 1 and _ID_LINE_RE.match(lines[0]):
                is_id_like = True

        looks_truncated = raw_str.endswith("...") or raw_str.endswith("..")

        if attempt < max_attempts and (not raw_str or is_id_like or looks_truncated):
            # increase budget and retry
            max_tokens += increment
            # update UX: inform user minimally by replacing the "Assistant: ..." line
            print("\r" + " " * 80 + "\r", end="", flush=True)
            print(f"Assistant: ... (retrying with token budget {max_tokens})", end="", flush=True)
            continue

        # If we reach here, either last attempt or heuristics didn't suggest retry.
        # Do one final conservative forced attempt with an explicit short-answer instruction
        forced_prompt = prompt + "\n\nAssistant: Please answer in one short sentence. If you don't know, reply exactly 'I don't know'."
        try:
            raw_forced = generate_text(llm, forced_prompt, max_tokens=max_tokens + increment)
            sanitized_forced = sanitize_assistant_text(raw_forced, last_user_message)
            if sanitized_forced:
                return sanitized_forced
        except Exception:
            pass

        # If still empty, break and return empty (caller will show fallback)
        break

    # final fallback: empty string
    return ""


def build_prompt(history, user_message):
    """Build simple conversation prompt with HISTORY_TURNS. Labels: Human / Assistant."""
    lines = [SYSTEM_PROMPT, "", "Conversation:"]
    hist = history[-HISTORY_TURNS*2:]
    for item in hist:
        if item.get("role") == "user":
            lines.append(f"Human: {item.get('text','').strip()}")
        else:
            lines.append(f"Assistant: {item.get('text','').strip()}")
    lines.append(f"Human: {user_message.strip()}")
    lines.append("Assistant:")
    return "\n".join(lines)


def chat_loop():
    print("\n=== Minimal Local Chat ===")
    print("Type 'exit' or 'quit' to stop.\n")
    history = []
    try:
        while True:
            # prompt the user
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye.")
                break

            # Build prompt and show a short progress indicator line
            prompt = build_prompt(history, user_input)
            display_prefix = "Assistant: ..."
            print(display_prefix, end='', flush=True)

            # Generate (blocking) and get a sanitized reply (auto-retries if needed)
            reply = generate_with_auto_extend(llm, prompt, last_user_message=user_input,
                                              initial_max_tokens=MAX_TOKENS, max_attempts=3, increment=150)
            if not reply:
                reply = "(no assistant reply produced)"

            # Overwrite the "Assistant: ..." line with the full reply
            full_line = "Assistant: " + reply
            clear_width = max(len(display_prefix), len(full_line))
            # clear line and carriage return, then print full reply
            print('\r' + ' ' * clear_width + '\r', end='', flush=True)
            print(full_line)
            print()  # blank line before next prompt

            # store history and continue (input() will print "You: " on next loop)
            history.append({"role": "user", "text": user_input})
            history.append({"role": "assistant", "text": reply})
    except KeyboardInterrupt:
        print("\nInterrupted. Bye.")


if __name__ == "__main__":
    chat_loop()
