import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
import os
from dotenv import load_dotenv
import json

load_dotenv()

FAISS_INDEX_PATH = "index/faiss.index"
DOCS_PATH = "index/documents.npy"

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index(FAISS_INDEX_PATH)
documents = np.load(DOCS_PATH, allow_pickle=True)

print("RAG system loaded")


def ask_rag(question):

    query_embedding = model.encode([question])

    distances, indices = index.search(query_embedding, k=5)

    context = "\n".join([documents[i] for i in indices[0]])

    prompt = f"""
You are an AI medical assistant.

Analyze the symptoms using the context.

Return the response ONLY in this JSON format:

{{
  "severity": "Mild | Moderate | Critical",
  "conditions": ["condition1", "condition2"],
  "advice": "short practical advice",
  "confidence": number_between_70_and_95
}}

Rules:
- Keep advice short (2 sentences)
- Only include the JSON response
- No explanations outside JSON

Context:
{context}

User symptoms:
{question}
"""



    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    text = response.text.strip()

    # Remove markdown formatting if Gemini adds it
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except:
        return {
            "severity": "Mild",
            "conditions": ["General illness"],
            "advice": text,
            "confidence": 80
        }