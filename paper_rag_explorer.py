import os
import sys
import pickle
import argparse
import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# Model Path Configurations
MINISTRAL_PATH = "models/Ministral-8B-Instruct-2410-Q4_K_M.gguf"
ROCKET_PATH = "models/rocket-3b.Q4_K_M.gguf"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # Lightweight and highly efficient local embedding model
CACHE_FILE = "rag_index_cache.pkl"
CHUNK_SIZE = 600  # Character count per chunk
CHUNK_OVERLAP = 150


def extract_chunks_from_pdfs(pdf_dir):
    """Reads all PDFs in a directory and splits them into chunks with metadata."""
    chunks = []
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(f"Error: No PDF files found in {pdf_dir}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF documents. Starting text extraction...")

    for filename in pdf_files:
        path = os.path.join(pdf_dir, filename)
        try:
            doc = fitz.open(path)
            document_text = ""
            for page in doc:
                document_text += page.get_text("text") + "\n"
            doc.close()

            # Simple character-based sliding window chunking
            start = 0
            while start < len(document_text):
                end = start + CHUNK_SIZE
                chunk_text = document_text[start:end].strip()
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "source": filename
                    })
                start += CHUNK_SIZE - CHUNK_OVERLAP
        except Exception as e:
            print(f"Warning: Could not parse {filename}. Skipping. Error: {e}")

    return chunks


def build_or_load_index(pdf_dir):
    """Loads cache if exists, otherwise generates embeddings and saves cache."""
    if os.path.exists(CACHE_FILE):
        print(f"Loading indexed documents from local cache: {CACHE_FILE}")
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)

    print("No cache found. Initializing embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    chunks = extract_chunks_from_pdfs(pdf_dir)
    if not chunks:
        print("Error: No text could be extracted from the directory.")
        sys.exit(1)

    print(f"Generating embeddings for {len(chunks)} text chunks. This may take a moment...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Normalize embeddings for easy cosine similarity (dot product)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / np.where(norms == 0, 1, norms)

    index_data = {
        "chunks": chunks,
        "embeddings": normalized_embeddings
    }

    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(index_data, f)
    print(f"Successfully indexed and cached data to {CACHE_FILE}")
    return index_data


def retrieve_context(query, index_data, embed_model, top_k=7):
    """Finds the most relevant chunks using normalized dot-product (cosine similarity)."""
    query_vector = embed_model.encode([query], convert_to_numpy=True)
    query_norm = np.linalg.norm(query_vector)
    if query_norm > 0:
        query_vector = query_vector / query_norm

    # Matrix multiplication gets all cosine similarities instantly
    similarities = np.dot(index_data["embeddings"], query_vector.T).squeeze()

    # Handle edge case if there's only 1 chunk total
    if similarities.ndim == 0:
        top_indices = [0]
    else:
        top_indices = np.argsort(similarities)[::-1][:top_k]

    retrieved_chunks = [index_data["chunks"][idx] for idx in top_indices]
    return retrieved_chunks


def main():
    parser = argparse.ArgumentParser(description="Simple Local RAG CLI for Scientific Papers")
    parser.add_argument("directory", help="Path to the directory containing your PDF files")
    parser.add_argument("--fast", action="store_true", help="Use Rocket-3B instead of Ministral-8B for speed")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist.")
        sys.exit(1)

    # Determine which model path to use
    selected_model_path = ROCKET_PATH if args.fast else MINISTRAL_PATH
    model_name = "Rocket-3B" if args.fast else "Ministral-8B"

    # 1. Build or Load Index
    index_data = build_or_load_index(args.directory)

    # 2. Re-instantiate the lightweight encoder for runtime queries
    print("Loading query encoder...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # 3. Load the Local GGUF LLM
    print(f"Loading {model_name} from {selected_model_path} (Context window: 4096)...")
    if not os.path.exists(selected_model_path):
        print(f"Error: Model file not found at {selected_model_path}. Please check your path assignments.")
        sys.exit(1)

    llm = Llama(
        model_path=selected_model_path,
        n_ctx=4096,
        n_threads=os.cpu_count() or 4
    )

    print("\n" + "=" * 50)
    print(f" RAG ASSISTANT READY ({model_name} Mode). Ask your questions.")
    print(" Type 'exit' or 'quit' to close.")
    print("=" * 50 + "\n")

    while True:
        try:
            query = input("\nUser Query: ").strip()
            if not query:
                continue
            if query.lower() in ['exit', 'quit']:
                break

            print("Retrieving context and generating answer...")

            # Retrieve chunks
            matched_chunks = retrieve_context(query, index_data, embed_model, top_k=7)

            # Format context explicitly showing source attribution
            context_str = ""
            for i, chunk in enumerate(matched_chunks, 1):
                context_str += f"--- CONTEXT BLOCK {i} (Source Document: {chunk['source']}) ---\n"
                context_str += f"{chunk['text']}\n\n"

            # Construct standard clean prompt layout
            prompt = (
                "System: You are an expert scientific research assistant. "
                "Analyze the provided context blocks extracted from local PDF documents to answer the user query. "
                "Every context block explicitly lists its 'Source Document' filename at the top.\n\n"
                f"CONTEXT:\n{context_str}\n"
                f"USER QUERY: {query}"
            )

            # Generate response
            response = llm(
                prompt,
                max_tokens=512,
                temperature=0.2,  # Low temp for factual extraction
                stop=["[/INST]", "</s>"]
            )

            answer = response["choices"][0]["text"].strip()
            print("\nAssistant Response:")
            print(answer)
            print("\n" + "-" * 40)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nAn error occurred during generation: {e}")


if __name__ == "__main__":
    main()
