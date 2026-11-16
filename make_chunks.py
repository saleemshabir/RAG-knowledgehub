from pathlib import Path
data_path = Path("data")
file_path = data_path / "input.txt"
text = file_path.read_text(encoding="utf-8")

print(text)

print("Total characters:", len(text))

def chunk_text(text, chunk_size=500):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks
chunks = chunk_text(text, chunk_size=500)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:\n{chunk}\n")