
import json
from pathlib import Path

import numpy as np
import requests


# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIRECTORY = PROJECT_ROOT

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"

TOP_K = 3
SIMILARITY_THRESHOLD = 0.20


def load_index(data_directory=None):
    """
    Load chunks and their embeddings from the specified index folder.
    """

    index_directory = Path(data_directory or DEFAULT_DATA_DIRECTORY)

    chunks_path = index_directory / "chunks.json"
    embeddings_path = index_directory / "embeddings.npy"

    if not chunks_path.exists() or not embeddings_path.exists():
        return [], np.empty((0, 384), dtype=np.float32)

    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    embeddings = np.load(embeddings_path, allow_pickle=False)

    if not isinstance(chunks, list):
        raise ValueError("chunks.json must contain a list.")

    if embeddings.ndim != 2:
        raise ValueError("embeddings.npy must contain a 2D matrix.")

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            "The number of chunks does not match the number of embeddings."
        )

    if embeddings.shape[1] != 384:
        raise ValueError(
            "Embedding dimensions do not match all-MiniLM-L6-v2."
        )

    return chunks, embeddings


def retrieve_chunks(
    query_text,
    model,
    data_directory=None,
    include_scores=False,
    verbose=False,
    top_k=None,
):
    """
    Retrieve the most relevant document chunks.

    When include_scores=True, returns:
        [(chunk_dictionary, similarity_score), ...]

    Otherwise, returns:
        [chunk_dictionary, ...]
    """

    query_text = query_text.strip()

    if not query_text:
        return []

    chunks, embeddings = load_index(data_directory)

    if not chunks:
        return []

    query_embedding = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )[0]

    if query_embedding.shape[0] != embeddings.shape[1]:
        raise ValueError(
            "Query embedding dimensions do not match the document index."
        )

    # Embeddings are normalized, so dot product gives cosine similarity.
    similarities = embeddings @ query_embedding

    ranked_indices = np.argsort(similarities)[::-1]

    results = []

    for index in ranked_indices:
        score = float(similarities[index])

        if score < SIMILARITY_THRESHOLD:
            continue

        chunk = chunks[int(index)]

        if include_scores:
            results.append((chunk, score))
        else:
            results.append(chunk)

        if len(results) >= (top_k or TOP_K):
            break

    if verbose:
        for chunk, score in (
            results if include_scores
            else [(item, 0.0) for item in results]
        ):
            print(
                f"Score: {score:.4f} | "
                f"Source: {chunk.get('source', 'Unknown')} | "
                f"Chunk: {chunk.get('chunk_id', 'Unknown')}"
            )

    return results


def build_prompt(question, relevant_chunks, detailed=False):
    """
    Build a prompt using only the retrieved document content.
    """

    context_parts = []

    for index, chunk in enumerate(relevant_chunks, start=1):
        source = chunk.get("source", "Unknown document")
        chunk_id = chunk.get("chunk_id", "Unknown")
        content = chunk.get("text", "")

        context_parts.append(
            f"[Source {index}: {source}, Chunk {chunk_id}]\n{content}"
        )

    context = "\n\n".join(context_parts)

    answer_instruction = (
        "Write a thorough, user-friendly explanation. Start with a direct answer, "
        "then organize useful details under clear headings. Include relevant "
        "background, how it works, practical uses, advantages, disadvantages or "
        "limitations, and examples when those details are supported by the "
        "document context. Connect related points found across the supplied "
        "chunks. If the document gives only limited information, explain what it "
        "does say and identify which requested aspects it does not cover. Do not "
        "pad the answer or add outside facts."
        if detailed
        else "Give a clear and concise answer."
    )

    return f"""
You are the AI assistant for KnowledgeHub.

Answer the user's question using the supplied document context as your source.

Rules:
1. Keep factual claims grounded in the context; do not fill gaps with outside knowledge.
2. If the answer or a requested detail is not available in the context, say so clearly.
3. Mention the relevant source document when appropriate.
4. Keep the answer accurate and easy to understand.
5. {answer_instruction}

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
""".strip()


def generate_answer(question, relevant_chunks, detailed=False):
    """
    Generate a complete answer using Ollama.
    """

    if not relevant_chunks:
        return "This information is not provided in the uploaded documents."

    prompt = build_prompt(
        question,
        relevant_chunks,
        detailed=detailed,
    )

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 900 if detailed else 256,
            },
        },
        timeout=(10, 180),
    )

    response.raise_for_status()

    result = response.json()

    answer = result.get("response", "").strip()

    if not answer:
        raise ValueError("Ollama returned an empty answer.")

    return answer


def open_answer_stream(question, relevant_chunks, detailed=False):
    """
    Open a streaming Ollama response.
    """

    prompt = build_prompt(
        question,
        relevant_chunks,
        detailed=detailed,
    )

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.2,
                "num_predict": 900 if detailed else 256,
            },
        },
        stream=True,
        timeout=(10, 180),
    )

    response.raise_for_status()

    return response


def iter_answer_tokens(response):
    """
    Read tokens from Ollama's newline-delimited JSON stream.
    """

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue

        if isinstance(line, bytes):
            line = line.decode("utf-8")

        try:
            data = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Could not decode Ollama response: {line}"
            ) from error

        if "error" in data:
            raise ValueError(data["error"])

        token = data.get("response", "")

        if token:
            yield token

        if data.get("done", False):
            break


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("KnowledgeHub document retrieval")
    print("Type 'exit' to stop.")

    while True:
        question = input("\nAsk a question: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        results = retrieve_chunks(
            question,
            model,
            data_directory=DEFAULT_DATA_DIRECTORY,
            verbose=True,
        )

        if not results:
            print("No relevant document chunks found.")
            continue

        print("\nRetrieved chunks:")

        for chunk in results:
            print(f"\nSource: {chunk.get('source', 'Unknown')}")
            print(chunk.get("text", ""))
