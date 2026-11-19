import json
import os
from functools import lru_cache
from pathlib import Path
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = BASE_DIR / "chunks.json"
EMBEDDINGS_FILE = BASE_DIR / "embeddings.npy"

TOP_K = 3
SIMILARITY_THRESHOLD = 0.20

OLLAMA_URL = os.environ.get(
    "OLLAMA_URL", "http://localhost:11434/api/generate"
)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "10m")
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "256"))


def load_data(data_directory=BASE_DIR):
    """Load the latest chunks and embeddings."""

    data_directory = Path(data_directory)
    chunks_file = data_directory / "chunks.json"
    embeddings_file = data_directory / "embeddings.npy"

    if not chunks_file.exists():
        raise FileNotFoundError(f"chunks.json not found in {data_directory}.")

    if not embeddings_file.exists():
        raise FileNotFoundError(f"embeddings.npy not found in {data_directory}.")

    return _load_data_cached(
        str(chunks_file.resolve()),
        str(embeddings_file.resolve()),
        chunks_file.stat().st_mtime_ns,
        embeddings_file.stat().st_mtime_ns,
    )


@lru_cache(maxsize=4)
def _load_data_cached(chunks_path, embeddings_path, chunks_modified, embeddings_modified):
    with Path(chunks_path).open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = np.load(embeddings_path)

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


def retrieve_chunks(
    query_text,
    model,
    data_directory=BASE_DIR,
    include_scores=False,
    verbose=True,
):
    """Retrieve relevant chunks using the latest data."""

    chunks, embeddings = load_data(data_directory)

    query_embedding = model.encode([query_text])

    if query_embedding.shape[1] != embeddings.shape[1]:
        raise ValueError(
            "Query and document embedding dimensions differ."
        )

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_count = min(TOP_K, len(similarities))
    top_indices = np.argpartition(similarities, -top_count)[-top_count:]
    ranked_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

    relevant_chunks = []

    if verbose:
        print("\n--- Retrieved Chunks ---")

    for index in ranked_indices:
        score = float(similarities[index])

        if score < SIMILARITY_THRESHOLD:
            break

        chunk = chunks[index]

        relevant_chunks.append((chunk, score) if include_scores else chunk)

        if verbose:
            print(f"\nChunk ID: {chunk['chunk_id']}")
            print(f"Source: {chunk['source']}")
            print(f"Similarity Score: {score:.4f}")
            print(chunk["text"])

        if len(relevant_chunks) >= TOP_K:
            break

    return relevant_chunks


def build_prompt(query_text, relevant_chunks):
    context = "\n\n".join(
        f"Source: {chunk['source']}, "
        f"Chunk ID: {chunk['chunk_id']}\n"
        f"{chunk['text']}"
        for chunk in relevant_chunks
    )

    prompt = f"""
Answer using only the supplied context, in up to three concise sentences.
If the answer is not in the context, say: I don't know based on the provided information.

Context:
{context}

Question:
{query_text}

Answer:
"""
    return prompt


def _generate_request(query_text, relevant_chunks, stream):
    return requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": build_prompt(query_text, relevant_chunks),
            "stream": stream,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {
                "num_predict": OLLAMA_NUM_PREDICT,
                "temperature": 0.2,
            },
        },
        stream=stream,
        timeout=(10, 120),
    )


def open_answer_stream(query_text, relevant_chunks):
    """Open Ollama's streamed response without waiting for generation to finish."""

    response = _generate_request(query_text, relevant_chunks, stream=True)
    response.raise_for_status()
    return response


def iter_answer_tokens(response):
    """Yield generated text fragments from an Ollama streaming response."""

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        data = json.loads(line)
        token = data.get("response")
        if isinstance(token, str) and token:
            yield token
        if data.get("done"):
            break


def generate_answer(query_text, relevant_chunks):
    """Generate an answer using Ollama."""

    response = _generate_request(query_text, relevant_chunks, stream=False)

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
