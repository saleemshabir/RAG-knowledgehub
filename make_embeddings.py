import numpy as np
import json
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the text file
data_path = Path("data")
file_path = data_path / "Saleem.txt"
text = file_path.read_text(encoding="utf-8")

# Generate chunks
def chunk_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

chunks = chunk_text(text)

print("Total chunks:", len(chunks))
embeddings = model.encode(chunks)
print("Embeddings generated successfully")
print("Shape of embeddings:", embeddings.shape)
np.save("embeddings.npy", embeddings)
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=4)
    