import chromadb
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
text = load_text("night_agent.txt")

def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks
chunks = chunk_text(text)

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

client = chromadb.Client()
collection = client.create_collection(name="rag_collection")

for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        embeddings=[embeddings[i].tolist()],
        ids=[str(i)]
    )

def retrieve(query, k=3):
    query_embedding = model.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k
    )
    return results["documents"][0]

query="Describe Rose Larkin and Peter Sutherland to me, in your own words"
retrieved_chunks = retrieve(query)
context = "\n\n".join(retrieved_chunks)
prompt = f"""
Answer the question using ONLY the context below.
Context:
{context}
Question:
{query}
"""

client=Groq(api_key=GROQ_API_KEY)
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "user", "content": prompt}
    ]
)
print(response.choices[0].message.content)