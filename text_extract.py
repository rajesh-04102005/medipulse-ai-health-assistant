import os
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

PDF_PATH = "data/DATASET.pdf"

INDEX_DIR = "index"

FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
DOCS_PATH = os.path.join(INDEX_DIR, "documents.npy")

os.makedirs(INDEX_DIR, exist_ok=True)


def extract_text(file):

    text = ""

    with open(file, "rb") as f:
        reader = PyPDF2.PdfReader(f)

        for page in reader.pages:
            content = page.extract_text()

            if content:
                text += content

    return text


def chunk_text(text, size=4):

    lines = text.split("\n")

    lines = [l.strip() for l in lines if l.strip()]

    chunks = []

    for i in range(0, len(lines), size):
        chunk = " ".join(lines[i:i+size])
        chunks.append(chunk)

    return chunks


model = SentenceTransformer("all-MiniLM-L6-v2")

text = extract_text(PDF_PATH)

documents = chunk_text(text)

embeddings = model.encode(documents)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

faiss.write_index(index, FAISS_INDEX_PATH)

np.save(DOCS_PATH, documents)

print("FAISS index created")