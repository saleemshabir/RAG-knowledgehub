
import json
from pathlib import Path

import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Project paths
BASE_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = BASE_DIR / "chunks.json"
EMBEDDINGS_FILE = BASE_DIR / "embeddings.npy"

# Retrieval settings
TOP_K = 3
SIMILARITY_THRESHOLD = 0.20

# Ollama settings
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"


def load_data():
    """Load the latest chunks and embeddings."""

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError("chunks.json not found.")

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError("embeddings.npy not found.")

    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = np.load(EMBEDDINGS_FILE)

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("No valid chunks found.")

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            "Number of chunks and embeddings do not match."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid values.")

    for chunk in chunks:
        if not all(
            key in chunk
            for key in ("chunk_id", "source", "text")
        ):
            raise ValueError("Invalid chunk metadata found.")

    return chunks, embeddings


def retrieve_chunks(query_text, model):
    """Retrieve relevant chunks using the latest data."""

    chunks, embeddings = load_data()

    query_embedding = model.encode([query_text])

    if query_embedding.shape[1] != embeddings.shape[1]:
        raise ValueError(
            "Query and document embedding dimensions differ."
        )

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    ranked_indices = np.argsort(similarities)[::-1]

    relevant_chunks = []

    print("\n--- Retrieved Chunks ---")

    for index in ranked_indices:
        score = float(similarities[index])

        if score < SIMILARITY_THRESHOLD:
            break

        chunk = chunks[index]

        relevant_chunks.append(chunk)

        print(f"\nChunk ID: {chunk['chunk_id']}")
        print(f"Source: {chunk['source']}")
        print(f"Similarity Score: {score:.4f}")
        print(chunk["text"])

        if len(relevant_chunks) >= TOP_K:
            break

    return relevant_chunks


def generate_answer(query_text, relevant_chunks):
    """Generate an answer using Ollama."""

    context = "\n\n".join(
        f"Source: {chunk['source']}, "
        f"Chunk ID: {chunk['chunk_id']}\n"
        f"{chunk['text']}"
        for chunk in relevant_chunks
    )

    prompt = f"""
You are a helpful assistant.
Answer using ONLY the provided context.
Do not use outside knowledge.

If the answer is not present in the context,
say: I don't know based on the provided information.

Context:
{context}

Question:
{query_text}

Answer:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    response_data = response.json()
    answer = response_data.get("response")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(
            "Ollama returned an empty or invalid answer."
        )

    return answer.strip()


def main():

    print("Loading RAG application...")

    print("\nLoading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("\nRAG Assistant is ready!")

    while True:

        try:
            query_text = input(
                "\nEnter your question (or type 'exit'): "
            )

        except (EOFError, KeyboardInterrupt):
            print("\nExiting RAG Assistant.")
            break

        if query_text.lower().strip() == "exit":
            print("Thank you for using the RAG assistant.")
            break

        if not query_text.strip():
            print("Please enter a question.")
            continue

        try:
            relevant_chunks = retrieve_chunks(
                query_text,
                model
            )

            if not relevant_chunks:
                print(
                    "\nI don't know based on the information provided."
                )
                continue

            answer = generate_answer(
                query_text,
                relevant_chunks
            )

            print("\n--- Generated Answer ---")
            print(answer)

            print("\n--- Sources ---")

            for chunk in relevant_chunks:
                print(
                    f"- {chunk['source']} | "
                    f"Chunk ID: {chunk['chunk_id']}"
                )

        except requests.ConnectionError:
            print(
                "\nError: Cannot connect to Ollama. "
                "Make sure Ollama is running."
            )

        except requests.Timeout:
            print("\nError: Ollama request timed out.")

        except requests.RequestException as error:
            print("\nOllama request failed:", error)

        except (ValueError, KeyError, TypeError) as error:
            print("\nProcessing error:", error)

        except Exception as error:
            print("\nUnexpected error:", error)


if __name__ == "__main__":

    try:
        main()

    except FileNotFoundError as error:
        print("\nFile error:", error)

    except (ValueError, json.JSONDecodeError) as error:
        print("\nData error:", error)

    except Exception as error:
        print("\nApplication error:", error)
