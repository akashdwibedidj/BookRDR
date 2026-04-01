import ollama
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer


model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path='./vector_db')



quastine = 'what are the top 7 habits'
embed_q = model.encode([quastine])
collection = client.get_or_create_collection(name='book_chunks')
results = collection.query(
   query_embeddings = [embed_q[0].tolist()],
   n_results=5
)
chunks = results['documents'][0]
metas  = results['metadatas'][0]
context = "\n\n".join(chunks)

prompt = f"""You are a helpful assistant. 
Using ONLY the following context from books, answer the question.
Do not use outside knowledge.

Context:
{context}

Question: {quastine}

Answer:"""

response = ollama.chat(
    model='mistral',
    messages = [{'role': 'user', 'content': prompt}]
)
print("\nAnswer")
print(response['message']['content'])
print("\nSources:")
for meta in metas:
    print(f" - {meta['book']}")