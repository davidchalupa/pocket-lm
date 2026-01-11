"""
Starter RAG script on a mini LM with progress bars for embedding + indexing.

 - small in-memory 'docs' list (just a few text notes)
 - embed docs with sentence-transformers
 - build FAISS index
 - retrieve top-k
 - call local Llama model via llama-cpp-python to generate an answer
"""

from sentence_transformers import SentenceTransformer
import faiss
from llama_cpp import Llama
from tqdm.auto import tqdm
import numpy as np
import os

# config
MODEL_PATH = "models/rocket-3b.Q4_K_M.gguf"  # <- update if your filename differs
EMBED_BATCH = 16      # sentences per encode() call
INDEX_ADD_CHUNK = 512 # vectors to add per faiss.add() step (tune for memory)
TOP_K = 2

# small helper: check model file
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at '{MODEL_PATH}'.\n"
        "If you've downloaded a GGUF to models/, update MODEL_PATH variable accordingly."
    )

# load small embedding model
# this is a sentence-transformers model: it maps sentences & paragraphs to a 384 dimensional dense
# vector space and can be used for tasks like clustering or semantic search
print("Loading embedding model (all-MiniLM-L6-v2)…")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# example docs (kept same as before)
docs = [
    {"id": "note1", "text": "How to bake a sourdough starter: feed flour and water daily."},
    {"id": "note2", "text": "Meeting notes: project timeline ends Q2; deliverables include API and UI."},
]

texts = [d["text"] for d in docs]
n_texts = len(texts)

# build embeddings with a progress bar (batching to show progress on large sets)
print(f"Encoding {n_texts} documents (batch_size={EMBED_BATCH}) …")
emb_batches = []
for i in tqdm(range(0, n_texts, EMBED_BATCH), desc="Embedding", unit="batch"):
    batch = texts[i : i + EMBED_BATCH]
    # we disable the internal progress bar of encode() and show our own
    batch_emb = embed_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
    emb_batches.append(batch_emb)

embs = np.vstack(emb_batches) if len(emb_batches) > 1 else emb_batches[0]
dim = embs.shape[1]
print(f"Embeddings shape: {embs.shape} (dim={dim})")

# create FAISS index and add embeddings in chunks with a progress bar
print("Building FAISS index …")
index = faiss.IndexFlatL2(dim)
n_vectors = embs.shape[0]
for i in tqdm(range(0, n_vectors, INDEX_ADD_CHUNK), desc="Indexing", unit="vecs"):
    index.add(embs[i : i + INDEX_ADD_CHUNK])

# query and retrieve top-k
query = "How often should I feed the starter?"
print(f"\nQuery: {query}")
q_emb = embed_model.encode([query], convert_to_numpy=True, show_progress_bar=False)
D, I = index.search(q_emb, k=min(TOP_K, n_vectors))
retrieved = [docs[i] for i in I[0]]

print("\nRetrieved context:")
for r in retrieved:
    print(f" - [{r['id']}] {r['text']}")

# build prompt with retrieved context (same as before)
context = "\n\n".join(f"[{r['id']}] {r['text']}" for r in retrieved)
prompt = (
    "You are a helpful assistant. Use the following context to answer the question.\n\n"
    f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
)

# run a small LLM with llama.cpp bindings (ensure model file .gguf path is correct)
print("\nLoading local LLM and generating answer …")
llm = Llama(model_path=MODEL_PATH)

print("\n")
print("--- Full prompt ---")
print(prompt)
print("--------------")

def generate_text(llm, prompt, max_tokens=150):
    # 1) Preferred high-level API: create_completion
    try:
        if hasattr(llm, "create_completion"):
            resp = llm.create_completion(prompt=prompt, max_tokens=max_tokens, stream=False)
            if isinstance(resp, dict):
                choices = resp.get("choices") or []
                if choices:
                    first = choices[0]
                    if isinstance(first, dict):
                        if "text" in first and first["text"]:
                            return first["text"].strip()
                        # chat style: message -> content
                        msg = first.get("message")
                        if isinstance(msg, dict):
                            content = msg.get("content") or msg.get("text")
                            if content:
                                return content.strip()
    except Exception:
        pass

    # 2) Older high-level API: create()
    try:
        if hasattr(llm, "create"):
            resp = llm.create(prompt=prompt, max_tokens=max_tokens)
            if isinstance(resp, dict):
                choices = resp.get("choices") or []
                if choices and isinstance(choices[0], dict) and "text" in choices[0]:
                    return choices[0]["text"].strip()
    except Exception:
        pass

    # 3) Low-level fallback: tokenize -> generate -> detokenize (works with older/low-level bindings)
    try:
        if hasattr(llm, "tokenize") and hasattr(llm, "generate") and hasattr(llm, "detokenize"):
            prompt_bytes = prompt.encode("utf-8")
            tokens = llm.tokenize(prompt_bytes)
            token_iter = llm.generate(tokens, max_tokens=max_tokens)
            pieces = []
            for tok in token_iter:
                try:
                    chunk = llm.detokenize([tok])
                    if isinstance(chunk, (bytes, bytearray)):
                        chunk = chunk.decode("utf-8", errors="ignore")
                    pieces.append(chunk)
                except Exception:
                    # some builds return small objects; ignore if detokenize fails
                    continue
            text = "".join(pieces).strip()
            if text:
                return text
    except Exception:
        pass

    # 4) Try common attribute-based extraction if a response object is present
    try:
        # some versions return an object with .generations
        if 'resp' in locals() and hasattr(resp, "generations"):
            return resp.generations[0][0].text.strip()
    except Exception:
        pass

    raise RuntimeError("No compatible generation API found on the Llama object.")

answer = generate_text(llm, prompt, max_tokens=150)

print("\n")
print("--- Answer ---")
print(answer)
print("--------------")
