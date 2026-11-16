import json
import numpy as np
import requests

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load text chunks
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Load embeddings
embeddings = np.load("embeddings.npy")

print("Chunks loaded:", len(chunks))
print("Embeddings loaded:", embeddings.shape)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Similarity threshold
SIMILARITY_THRESHOLD = 0.05

# Ask multiple questions
while True:

    query_text = input("\nEnter your question (or type 'exit'): ")

    if query_text.lower().strip() == "exit":
        print("Thank you for using the RAG application!")
        break

    if not query_text.strip():
        print("Please enter a question.")
        continue

    # Convert question into embedding
    query_embedding = model.encode([query_text])

    # Calculate cosine similarity
    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )

    print("Similarities:", similarities)

    # Find best matching chunk
    best_match_index = np.argmax(similarities)
    best_score = similarities[0][best_match_index]

    print("Similarity score:", best_score)

    # Check relevance
    if best_score < SIMILARITY_THRESHOLD:
        print("I couldn't find relevant information in your documents.")
        continue

    best_chunk = chunks[best_match_index]
    print("Best matching chunk:", best_chunk)

    # Prepare prompt
    prompt = f"""
You are a helpful assistant.
Answer using ONLY the provided context.
If the answer is not present in the context, say you don't know.

Context:
{best_chunk}

Question:
{query_text}

Answer:
"""

    # Generate answer using Ollama
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:1.5b",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()
        answer = response.json()["response"]

        print("\nGenerated Answer:")
        print(answer)

    except requests.RequestException as e:
        print("Error connecting to Ollama:", e)