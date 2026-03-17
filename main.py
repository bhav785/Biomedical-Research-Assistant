import uuid

from Bio import Entrez
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from groq import Groq
import os
import chromadb

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
Entrez.email = "yourname@gmail.com"

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Chroma (persistent)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="papers",
    metadata={"hnsw:space": "cosine"}
)

def fetch_pubmed_abstracts(query, max_results=20):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    ids = record["IdList"]

    abstracts = []

    for pubmed_id in ids:
        fetch_handle = Entrez.efetch(db="pubmed", id=pubmed_id, rettype="abstract", retmode="text")
        text = fetch_handle.read()
        abstracts.append(text)

    return abstracts


def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

def clean_text(text):
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        if "doi:" in line.lower():
            continue
        if "author information" in line.lower():
            continue
        cleaned.append(line)

    return " ".join(cleaned)
def store_in_chroma(chunks):
    embeddings = model.encode(chunks)

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i].tolist()],
            ids=[f"{i}_{uuid.uuid4()}"]
        )

def search_chroma(query, top_k=5):
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    return results["documents"][0]
def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a biomedical research assistant.

STRICT INSTRUCTIONS:
- Answer ONLY using the provided context
- If drug names are mentioned, you MUST list them
- Do NOT say "not mentioned" if any drug appears in the context
- Be specific and concise

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content

# STEP 1: Fetch data
data = fetch_pubmed_abstracts("drug repurposing cancer therapy", 20)

# STEP 2: Clean + chunk
all_chunks = []
for paper in data:
    cleaned = clean_text(paper)
    chunks = chunk_text(cleaned)
    all_chunks.extend(chunks)

print(f"Total chunks: {len(all_chunks)}")

# STEP 3: Store in vector DB
store_in_chroma(all_chunks)

# STEP 4: Query
query = input("Enter query: ")

retrieved_chunks = search_chroma(query)

# print("\nRetrieved Chunks:\n")
# for chunk in retrieved_chunks:
#     print("-", chunk[:200])

# STEP 5: Generate answer
answer = generate_answer(query, retrieved_chunks)

print("\nFINAL ANSWER:\n")
print(answer)