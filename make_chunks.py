from pathlib import Path

# Step 1: Set the data folder path
data_path = Path("data")

# Step 2: Set the text file path
file_path = data_path / "Saleem.txt"

# Step 3: Read the text file
text = file_path.read_text(encoding="utf-8")

# Step 4: Print the original text
print(text)

# Step 5: Print the total number of characters
print("Total characters:", len(text))


# Step 6: Create the chunking function
def chunk_text(text, chunk_size=500):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks


# Step 7: Generate chunks
chunks = chunk_text(text, chunk_size=500)

# Step 8: Display each chunk
for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:\n{chunk}\n")